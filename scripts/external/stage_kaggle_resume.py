"""Stage the Kaggle Resume Dataset (snehaanbhawal, LiveCareer resumes): raw/ -> cvs.parquet.

    python -I scripts/external/stage_kaggle_resume.py <raw dir> <staged dir>

One row per Resume.csv row. `text` is `Resume_str` stripped and nothing else: LiveCareer's
space padding is kept, because whether to collapse it (or re-render from `Resume_html`, which
`extra` keeps) is a pool decision, not a staging one. `occupation` is the uploader's Category.

`extra.near_dup_cluster` groups CVs whose word 5-gram shingle sets have Jaccard >= 0.8. The
acquirer found them with MinHash LSH, which missed 3 of the 6 pairs an exhaustive pass finds;
the verifier patched those 6 rows (`_scripts/kaggle_resume/verify/v10_fix_near_dup_cluster.py`).
Here the pairs come from the exhaustive pass directly, so nothing depends on LSH luck. A cluster
is named after the member with the larger row index, the acquirer's union-find convention.
The FAIRE overlap is a reason to get this right: FAIRE's 98 base CVs are Kaggle CVs, and the
external holdout drops them from any training pool by text.

Reads raw/extracted/Resume/Resume.csv and lists raw/extracted/data/data/<CATEGORY>/<ID>.pdf
(only the file names are used; no PDF is opened).
"""

from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import dumps, main, marker_patterns, write_table  # noqa: E402

SLUG = "kaggle_resume"
TABLES = ("cvs",)
REQUIRES = ()
CSV_REL = "Resume/Resume.csv"
NEAR_DUP_JACCARD = 0.8


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def headline_from_html(h: str) -> str:
    """The LiveCareer headline span (id '<n>LNAM<m>'), which names the role better than Category."""
    m = re.search(r'<span\b[^>]*\bid="\d+LNAM\d*"[^>]*>(.*?)</span>', h, re.S)
    return norm_ws(html.unescape(re.sub(r"<[^>]+>", " ", m.group(1)))) if m else ""


def shingles(text: str, k: int = 5) -> frozenset[str]:
    toks = re.findall(r"\w+", text.lower())
    return frozenset(" ".join(toks[i:i + k]) for i in range(max(1, len(toks) - k + 1)))


def near_dup_pairs(texts: list[str], threshold: float = NEAR_DUP_JACCARD) -> list[tuple[int, int]]:
    """Every (i, j), i < j, with shingle Jaccard >= threshold, by exhaustive comparison.

    Sorting by set size lets the inner loop stop as soon as the size ratio alone rules a
    match out (Jaccard <= |small| / |large|), which keeps 2,484 CVs to a few seconds.
    """
    sets = [shingles(t) for t in texts]
    order = sorted(range(len(sets)), key=lambda i: len(sets[i]))
    pairs = []
    for pos, i in enumerate(order):
        si, li = sets[i], len(sets[i])
        for j in order[pos + 1:]:
            lj = len(sets[j])
            if li < threshold * lj:
                break
            inter = len(si & sets[j])
            if inter and inter / (li + lj - inter) >= threshold:
                pairs.append((min(i, j), max(i, j)))
    return sorted(pairs)


def cluster_ids(ids: list[str], pairs: list[tuple[int, int]]) -> list[str]:
    """Union-find over the pairs; each clustered row gets its root's id, every other row ''."""
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    sizes: dict[int, int] = {}
    for i in range(len(ids)):
        sizes[find(i)] = sizes.get(find(i), 0) + 1
    return [ids[find(i)] if sizes[find(i)] > 1 else "" for i in range(len(ids))]


def pdf_paths(pdf_root: Path) -> dict[str, str]:
    """ID -> 'data/data/<CATEGORY>/<ID>.pdf' from the folder listing."""
    cat_of = {}
    for cat in sorted(os.listdir(pdf_root)):
        d = pdf_root / cat
        if d.is_dir():
            for f in os.listdir(d):
                cat_of[os.path.splitext(f)[0]] = cat
    return {i: f"data/data/{c}/{i}.pdf" for i, c in cat_of.items()}


def stage(raw: Path, out: Path) -> dict[str, int]:
    extracted = raw / "extracted"
    df = pd.read_csv(extracted / CSV_REL, dtype=str, keep_default_na=False)
    text = df["Resume_str"].str.strip()
    ids = df["ID"].astype(str).tolist()
    clusters = cluster_ids(ids, near_dup_pairs(text.tolist()))
    pdf_rel = pdf_paths(extracted / "data" / "data")
    rows = []
    for k, r in enumerate(df.to_dict("records")):
        t = text.iat[k]
        pats = marker_patterns(t)
        extra = {
            "Category": r["Category"],
            "headline": headline_from_html(r["Resume_html"]),
            "char_len": len(t),
            "in_400_6000": 400 <= len(t) <= 6000,
            "protected_marker_hit": bool(pats),
            "protected_marker_patterns": pats,
            "near_dup_cluster": clusters[k],
            "pdf_path": pdf_rel.get(r["ID"]),
            "Resume_html": r["Resume_html"],
        }
        rows.append({"cv_id": str(r["ID"]), "text": t, "lang": "en", "occupation": r["Category"],
                     "base_cv_id": str(r["ID"]), "variant": "", "split": "",
                     "source_file": CSV_REL, "extra": dumps(extra)})
    cvs = pd.DataFrame(rows)
    write_table(cvs, out / "cvs.parquet")
    return {"cvs": len(cvs), "near_dup_rows": sum(1 for c in clusters if c)}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
