"""Stage ResumeAtlas (13,389 OCR'd resumes, 43 categories): raw/ -> cvs.parquet.

    data-tools-venv/bin/python -I scripts/external/stage_resumeatlas.py <raw dir> <staged dir>

The text comes from the least-processed release, the GitHub LFS Dataset.csv. The HF copy
(Preprocessed_Data.csv, also train.csv on HF) is lowercased, stopword-stripped and
punctuation-free, which is fine for a classifier and useless as a CV, so it is read only to
check that its rows line up with the raw ones (`extra.hf_preprocessed_token_containment`).
`text` keeps the OCR export header (a U+FEFF and an underscore rule, which `strip` leaves).

The source has no ids, so `cv_id` is the row index (`resumeatlas-r00042`): stable only for the
LFS object whose sha256 the manifest pins. Every per-row flag in `extra` is computed here, in
one pass over the raw text, because each is a filter a pool may want and they must agree with
each other:

* `inferred_source`: the file is three blocks of 43 alphabetical category runs whose sizes
  equal the README's Bing/Google/LiveCareer counts, so a row's block names its scrape source;
* `exact_dup_of` / `near_dup_cluster_j08` / `near_dup_keep_j08`: exact copies, and word
  5-shingle Jaccard >= 0.8 clusters on alnum-normalised text (MinHash LSH at 0.7 for
  candidates, exact Jaccard to confirm; the representative is the lowest row index). LSH
  misses some pairs: an exhaustive pass (2026-10-19) clusters 3,421 rows instead of 3,400
  and would flip `near_dup_keep_j08` on 14. LSH is kept so a re-stage reproduces the
  verified file; switching is a decision, see kaggle_resume, which made it;
* OCR quality (`rare_token_share`, `glued_alnum_per_1k_tokens`, `poor_ocr_flag`), langdetect
  (seed 0), PII presence and placeholder detection, the LiveCareer 'Jessica Claire' pseudonym,
  lorem-ipsum templates, and whether the category's keyword appears in the text.

The acquirer computed these in a profiling script and staged from its output; the two steps are
merged here. Needs datasketch and langdetect.
"""

from __future__ import annotations

import collections
import hashlib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import dumps, main, marker_patterns, require, write_table  # noqa: E402

SLUG = "resumeatlas"
TABLES = ("cvs",)
REQUIRES = ("datasketch", "langdetect")
RAW_CSV = "github_lfs/Dataset.csv"
PRE_CSV = "github_lfs/Preprocessed_Data.csv"

README_SOURCES = {2722: "bing_images", 3015: "google_images", 7652: "livecareer"}
SOURCE_METHOD = ("file-order block (3 blocks x 43 alphabetical category runs; block sizes "
                 "2722/3015/7652 equal README Bing/Google/LiveCareer counts)")
EXPORT_HEADER = "﻿" + "_" * 16
LOREM = re.compile(r"(?i)\blorem\b|\bipsum\b|dolor sit amet|consectetur|\bmaecenas\b|\baenean\b|"
                   r"\bvestibulum\b")
GLUED_NAME = re.compile(r"[a-z]Jessica|Jessica[a-z]|[a-z]Claire|Claire[a-z]")
WORD = re.compile(r"[A-Za-z]+")
#: letters and digits glued in one token ('l0ve', '0ffice'), a typical OCR misread
GLUE = re.compile(r"\b(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{4,}\b")
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{2,4}\)|\d{3})[\s.\-]?\d{3}[\s.\-]?\d{3,4}")
URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+|\b[\w\-]+\.(?:com|net|org|io|in)/\S*")
PLACEHOLDER_EMAIL = re.compile(
    r"(?i)(example\.com|resumesample@|email@|yourname|your\.?name|name@|xxx|somedomain|"
    r"resumeworded|@email\.com|@domain\.com|first\.last|john\.?doe)")
PLACEHOLDER_PHONE = re.compile(
    r"555[\s.\-)]*\d{3}[\s.\-]?\d{4}|\(555\)|123[\s.\-]?456[\s.\-]?7890|000[\s.\-]?000[\s.\-]?0000|"
    r"xxx")
HEADER_WORDS = {"resume", "curriculum", "vitae", "summary", "profile", "professional",
                "experience", "education", "skills", "objective", "contact", "personal", "career",
                "work", "history", "about", "me", "cv", "details", "information", "qualifications",
                "highlights", "overview", "page", "of", "and"}
#: Category -> a keyword its CVs should mention; a weak label-consistency check.
CAT_KEYS = {
    "Accountant": r"account", "Advocate": r"advocate|attorney|lawyer|legal|law\b",
    "Agriculture": r"agricult|farm|crop", "Apparel": r"apparel|fashion|garment|clothing|retail",
    "Architecture": r"architect", "Arts": r"\bart|artist|museum|design",
    "Automobile": r"automo|vehicle|car\b|auto\b",
    "Aviation": r"aviation|aircraft|airline|flight|pilot", "Banking": r"bank",
    "Blockchain": r"blockchain|crypto|ethereum|solidity",
    "BPO": r"bpo|call center|customer service|outsourc",
    "Building and Construction": r"construct|building", "Business Analyst": r"business analy",
    "Civil Engineer": r"civil", "Consultant": r"consult",
    "Data Science": r"data scien|machine learning|data analy", "Database": r"database|dba|sql",
    "Designing": r"design", "DevOps": r"devops|ci/cd|kubernetes|jenkins|docker",
    "Digital Media": r"digital|media|social media|marketing",
    "DotNet Developer": r"\.net|dotnet|c#|asp", "Education": r"teach|educat|school|tutor",
    "Electrical Engineering": r"electric", "ETL Developer": r"\betl\b|informatica|data warehouse",
    "Finance": r"financ", "Food and Beverages": r"food|beverage|restaurant|chef|cook|culinary",
    "Health and Fitness": r"health|fitness|trainer|wellness",
    "Human Resources": r"human resource|\bhr\b|recruit",
    "Information Technology": r"information technology|\bit\b|network|technical support",
    "Java Developer": r"java", "Management": r"manag", "Mechanical Engineer": r"mechanical",
    "Network Security Engineer": r"security|network", "Operations Manager": r"operation",
    "PMO": r"\bpmo\b|project manag|program manag",
    "Public Relations": r"public relation|\bpr\b|communications", "Python Developer": r"python",
    "React Developer": r"react", "Sales": r"sales|sell", "SAP Developer": r"\bsap\b",
    "SQL Developer": r"sql", "Testing": r"test|\bqa\b|quality assurance", "Web Designing": r"web",
}
NEAR_DUP_JACCARD = 0.8
LSH_THRESHOLD, NUM_PERM, SHINGLE = 0.7, 128, 5


def cid(i: int) -> str:
    return f"resumeatlas-r{int(i):05d}"


def inferred_source(categories: pd.Series) -> pd.Series:
    runs = (categories != categories.shift()).cumsum()
    first_cat = categories.groupby(runs).first()
    block = runs.map((first_cat == "Accountant").cumsum())  # a block restarts at 'Accountant'
    sizes = block.value_counts().sort_index().to_dict()
    return block.map({b: README_SOURCES.get(n, f"block{b}") for b, n in sizes.items()})


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t).lower()
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def shingles(t: str) -> set[str]:
    w = t.split()
    if len(w) < SHINGLE:
        return {" ".join(w)} if w else {""}
    return {" ".join(w[i:i + SHINGLE]) for i in range(len(w) - SHINGLE + 1)}


def near_dup_roots(normed: list[str]) -> list[int]:
    """Union-find root (lowest row index) of each row's Jaccard >= 0.8 cluster."""
    from datasketch import MinHash, MinHashLSH

    sh = [shingles(t) for t in normed]
    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    mhs = []
    for i, s in enumerate(sh):
        m = MinHash(num_perm=NUM_PERM, seed=1)
        m.update_batch([x.encode("utf-8") for x in s])
        mhs.append(m)
        lsh.insert(str(i), m)
    parent = list(range(len(sh)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, m in enumerate(mhs):
        for j in map(int, lsh.query(m)):
            if j > i and len(sh[i] & sh[j]) / max(1, len(sh[i] | sh[j])) >= NEAR_DUP_JACCARD:
                a, b = find(i), find(j)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(len(sh))]


def first_line_looks_like_name(t: str) -> bool:
    """First non-empty line is 2-4 capitalised alphabetic words and none is a header word."""
    for ln in t.splitlines():
        s = ln.strip()
        if not s:
            continue
        toks = s.split()
        return (2 <= len(toks) <= 4
                and all(re.fullmatch(r"[A-Za-z][A-Za-z'\-\.]*", w) for w in toks)
                and all(w[0].isupper() for w in toks)
                and not any(w.lower().strip(".") in HEADER_WORDS for w in toks))
    return False


def language(t: str) -> tuple[str, float]:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    try:
        r = detect_langs(t[:3000])
        return r[0].lang, float(r[0].prob)
    except Exception:  # langdetect raises on text with no detectable features
        return "unk", 0.0


def containment(raw_text: str, pre_text: str) -> float:
    """Share of the preprocessed row's 3+ letter tokens found in the raw row."""
    ta = set(re.findall(r"[a-z]{3,}", raw_text.lower()))
    tb = set(re.findall(r"[a-z]{3,}", pre_text.lower()))
    return len(tb & ta) / max(1, len(tb))


def stage(raw: Path, out: Path) -> dict[str, int]:
    require(*REQUIRES)
    df = pd.read_csv(raw / RAW_CSV, dtype=str, keep_default_na=False, encoding="utf-8")
    pre = pd.read_csv(raw / PRE_CSV, dtype=str, keep_default_na=False, encoding="utf-8")
    if len(df) != len(pre) or not (df["Category"].values == pre["Category"].values).all():
        raise ValueError("Dataset.csv and Preprocessed_Data.csv rows no longer line up")
    text = df["Text"].str.strip()
    body = text.str.replace(r"^﻿_{5,}\s*", "", regex=True)  # analysis only; text keeps it
    source = inferred_source(df["Category"])

    tokens = body.map(lambda t: [w.lower() for w in WORD.findall(t)])
    doc_freq: collections.Counter = collections.Counter()
    for toks in tokens:
        doc_freq.update(set(toks))
    sha1 = text.map(lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest())
    first_row_of_sha = {}
    for i, s in enumerate(sha1):
        first_row_of_sha.setdefault(s, i)
    group_size = sha1.map(sha1.value_counts())
    roots = near_dup_roots(text.map(norm).tolist())
    cluster_size = collections.Counter(roots)

    rows = []
    for i in range(len(df)):
        t, b, toks = text.iat[i], body.iat[i], tokens.iat[i]
        n_tok = max(1, len(toks))
        rare = sum(1 for w in toks if doc_freq[w] <= 2 and len(w) > 2) / n_tok
        glued = 1000 * len(GLUE.findall(b)) / max(1, len(toks))
        letter_share = sum(1 for ch in b if ch.isalpha()) / max(1, len(b))
        lang, prob = language(b)
        emails, phones = EMAIL.findall(t), PHONE.findall(t)
        root = roots[i]
        clustered = cluster_size[root] > 1
        exact_rep = first_row_of_sha[sha1.iat[i]]
        extra = {
            "row_index": i,
            "category": df.at[i, "Category"],
            "inferred_source": source.iat[i],
            "inferred_source_method": SOURCE_METHOD,
            "text_sha1": sha1.iat[i],
            "exact_dup_group_size": int(group_size.iat[i]),
            "exact_dup_of": None if exact_rep == i else cid(exact_rep),
            "near_dup_cluster_j08": cid(root) if clustered else None,
            "near_dup_keep_j08": (root == i) or not clustered,
            "protected_marker_hits_en": marker_patterns(t),
            "len_chars": len(t),
            "has_email": bool(emails),
            "email_all_placeholder": bool(emails) and all(PLACEHOLDER_EMAIL.search(e)
                                                          for e in emails),
            "has_phone_like": bool(phones),
            "phone_all_placeholder": bool(phones) and all(PLACEHOLDER_PHONE.search(p)
                                                          for p in phones),
            "has_url": bool(URL.search(t)),
            "first_line_looks_like_name": first_line_looks_like_name(b),
            "livecareer_jessica_claire_pseudonym": bool(re.search(r"(?i)jessica\s+claire", t)),
            "pseudonym_substring_artifact": bool(GLUED_NAME.search(t)),
            "lorem_ipsum_template": bool(LOREM.search(t)),
            "ocr_export_header": df.at[i, "Text"].startswith(EXPORT_HEADER),
            "langdetect": lang, "langdetect_prob": round(prob, 4),
            "rare_token_share": round(rare, 4),
            "glued_alnum_per_1k_tokens": round(glued, 2),
            "poor_ocr_flag": rare > 0.10 or glued > 20 or letter_share < 0.55,
            "category_keyword_in_text": bool(re.search(CAT_KEYS[df.at[i, "Category"]], t,
                                                       flags=re.I)),
            "hf_preprocessed_row_index": i,
            "hf_preprocessed_token_containment": round(containment(t, pre.at[i, "Text"].strip()),
                                                       4),
        }
        rows.append({"cv_id": cid(i), "text": t, "lang": "en", "occupation": df.at[i, "Category"],
                     "base_cv_id": cid(i), "variant": "", "split": "",
                     "source_file": f"raw/{RAW_CSV}", "extra": dumps(extra)})
    cvs = pd.DataFrame(rows)
    write_table(cvs, out / "cvs.parquet")
    return {"cvs": len(cvs), "near_dup_rows": sum(1 for r in roots if cluster_size[r] > 1)}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
