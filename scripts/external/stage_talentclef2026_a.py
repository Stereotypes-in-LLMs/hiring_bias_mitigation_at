"""Stage TalentCLEF 2026 Task A (Zenodo 19652670, v0.3.0): raw/ -> cvs, jobs, pairs parquet.

    python -I scripts/external/stage_talentclef2026_a.py <raw dir> <staged dir>

Task A is CV retrieval for a job query, in English and Spanish, with a development and a test
split. The release does not say so, but every CV has a sibling: two LLM fillings of one
unreleased masked resume that differ in gender (and with it the name, contact details and
origin-linked entities). That pairing is what makes the set a gender counterfactual, so it is
reconstructed here rather than left to each consumer:

* siblings are mutual nearest neighbours by word 5-gram Jaccard, within one split and
  language; the lower numeric `c_id` of the two is the designated base;
* each sibling's gender is inferred from Spanish o/a and -or/-ora alternations between the two
  ES texts, and carried to EN through an EN<->ES alignment on language-invariant tokens
  (e-mail addresses, LinkedIn handles, digits);
* EN<->ES JD alignment uses identical qrels after CV alignment on dev, and paragraph/bullet
  structure (length ratio as tie-break) on test, which has no public qrels.

The base role is chosen per language, so for about half the underlying pairs the EN base and
the ES base are opposite siblings; cross-lingual work joins on `extra.aligned_*_cv_id`. Pairs
are the dev qrels (positives only; the test qrels are not public). Reads
raw/extracted/TaskA/{development,test}/{en,es}/{queries,corpus}/<id> and the dev qrels.tsv.
"""

from __future__ import annotations

import collections
import difflib
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import main, write_table  # noqa: E402

SLUG = "talentclef2026_a"
TABLES = ("cvs", "jobs", "pairs")
REQUIRES = ()
SPLITS = {"development": "dev", "test": "test"}
LANGS = ["en", "es"]
GENDER_HOW = "es o/a morphology between siblings; en via en-es alignment"
BASE_RULE = ("lower numeric c_id of the two LLM-filled siblings (both derive from one unreleased "
             "masked resume)")
JD_ALIGN_HOW = ("dev: identical qrels after CV alignment; test: paragraph/bullet structure "
                "(+length tie-break)")
QRELS_NOTE = ("qrels lists positives only; CV-JD combinations not listed are "
              "unjudged/implicitly non-relevant for MAP")

INV = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+|\d+")
WORD = re.compile(r"\w+", re.U)
TITLE_PATTERNS = [
    re.compile(r"^(?:Job Title|Position|Título del Puesto|Puesto)\s*:\s*(.+)$", re.I),
    re.compile(r"^We are seeking (?:an?|the) (.+?) to join\b", re.I),
    re.compile(r"^Estamos buscando (?:un/a|una?|un\(a\)) (.+?) para (?:unirse|incorporarse|"
               r"sumarse)\b", re.I),
]
OVERVIEW_PATTERNS = [
    re.compile(r"(?:seeking|looking for) (?:an?|the) (.+?)(?: to | who |,|\.)", re.I),
    re.compile(r"(?:buscando|busca) (?:un/a|una?|un) (.+?)(?: para | que |,|\.)", re.I),
]


def load_dir(d: Path) -> dict[str, str]:
    """id -> text for one queries/ or corpus/ folder, in numeric id order; strict UTF-8."""
    return {f.name: f.read_bytes().decode("utf-8")
            for f in sorted(d.iterdir(), key=lambda p: int(p.name))}


def shingles(t: str, k: int = 5) -> set[str]:
    w = re.findall(r"\w+", t.lower())
    return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}


def mutual_nn_pairs(docs: dict[str, str]) -> dict[str, tuple[str, float]]:
    """id -> (sibling id, Jaccard) for every document whose nearest neighbour is mutual."""
    sh = {i: shingles(t) for i, t in docs.items()}
    best: dict[str, tuple[float, str]] = {}
    for a, b in itertools.combinations(list(docs), 2):
        j = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
        if j > best.get(a, (-1, None))[0]:
            best[a] = (j, b)
        if j > best.get(b, (-1, None))[0]:
            best[b] = (j, a)
    pairs = {}
    for a, (j, b) in best.items():
        if best[b][1] == a and a not in pairs:
            pairs[a] = (b, j)
            pairs[b] = (a, j)
    return pairs


def cjac(a: collections.Counter, b: collections.Counter) -> float:
    u = sum((a | b).values())
    return sum((a & b).values()) / u if u else 0.0


def align_cvs(en: dict[str, str], es: dict[str, str]) -> dict[str, str]:
    """EN id -> ES id where each is the other's best match on invariant tokens."""
    se = {k: collections.Counter(INV.findall(v)) for k, v in en.items()}
    ss = {k: collections.Counter(INV.findall(v)) for k, v in es.items()}
    best_en = {a: sorted(((cjac(se[a], ss[b]), b) for b in es), reverse=True)[0][1] for a in en}
    best_es = {b: max((cjac(se[a], ss[b]), a) for a in en)[1] for b in es}
    return {a: b for a, b in best_en.items() if best_es[b] == a}


def jd_struct(t: str) -> tuple:
    blocks = re.split(r"\n\s*\n", t.strip())
    bullets = [len([ln for ln in b.splitlines() if re.match(r"^\s*[-•*·]\s", ln)]) for b in blocks]
    return (len(blocks), tuple(bullets))


def align_jds(en: dict[str, str], es: dict[str, str]) -> dict[str, str]:
    """EN JD id -> ES JD id by paragraph/bullet layout; ambiguous ones by length ratio."""
    se = {k: jd_struct(v) for k, v in en.items()}
    ss = {k: jd_struct(v) for k, v in es.items()}
    final: dict[str, str] = {}
    ambiguous: list[tuple[str, list[str]]] = []
    for a in en:
        cands = [b for b in es if ss[b] == se[a]]
        if len(cands) == 1:
            final[a] = cands[0]
        else:
            ambiguous.append((a, cands))
    ratios = [len(es[b]) / len(en[a]) for a, b in final.items()]
    med = statistics.median(ratios) if ratios else 1.15
    for a, cands in ambiguous:
        taken = set(final.values())
        cands = [c for c in cands if c not in taken] or [b for b in es if b not in taken]
        final[a] = min(cands, key=lambda b, a=a: abs(len(es[b]) / len(en[a]) - med))
    return final


def gender_votes(ta: str, tb: str) -> tuple[int, int]:
    """(votes that a is female, votes that a is male) from Spanish gendered word alternations."""
    wa, wb = WORD.findall(ta), WORD.findall(tb)
    sm = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    fem = masc = 0

    def feminine_vs(x: str, y: str) -> bool:
        return ((x[:-1] == y[:-1] and x[-1] == "a" and y[-1] == "o")
                or (x == y + "a" and y.endswith(("or", "ón", "és")))
                or (x[:-2] == y[:-1] and x.endswith("as") and y.endswith("os"))
                or (x in ("la", "una", "ella") and y in ("el", "un", "él")))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue
        for x, y in zip(wa[i1:i2], wb[j1:j2]):
            xl, yl = x.lower(), y.lower()
            if len(xl) < 3 or len(yl) < 3:
                continue
            if feminine_vs(xl, yl):
                fem += 1
            elif feminine_vs(yl, xl):
                masc += 1
    return fem, masc


def extract_title(text: str) -> tuple[str, str, str]:
    """(title, how it was found, first line) of a JD."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    first = lines[0]
    header = first.rstrip(":").lower()
    cands = [first] + (lines[1:4] if header in ("overview", "descripción general",
                                                "job description",
                                                "descripción del puesto") else [])
    for ln in cands:
        for p in TITLE_PATTERNS:
            m = p.search(ln)
            if m:
                return m.group(1).strip().rstrip("."), "pattern", first
    if header in ("overview", "descripción general"):
        for ln in lines[1:4]:
            for p in OVERVIEW_PATTERNS:
                m = p.search(ln)
                if m:
                    return m.group(1).strip(), "overview_sentence", first
    return first, "first_line", first


def stage(raw: Path, out: Path) -> dict[str, int]:
    root = raw / "extracted" / "TaskA"
    docs = {}
    for sp in SPLITS:
        for lg in LANGS:
            for kind in ("queries", "corpus"):
                docs[(sp, lg, kind)] = load_dir(root / sp / lg / kind)
    qrels = {lg: [ln.split("\t") for ln in (root / "development" / lg / "qrels.tsv")
                  .read_bytes().decode("utf-8").splitlines() if ln.strip()] for lg in LANGS}
    qset = {lg: collections.defaultdict(set) for lg in LANGS}
    pos_by_cv = {lg: collections.defaultdict(list) for lg in LANGS}
    for lg in LANGS:
        for r in qrels[lg]:
            qset[lg][r[0]].add(r[2])
            pos_by_cv[lg][r[2]].append(r[0])

    siblings = {(sp, lg): mutual_nn_pairs(docs[(sp, lg, "corpus")])
                for sp in SPLITS for lg in LANGS}
    align_cv = {sp: align_cvs(docs[(sp, "en", "corpus")], docs[(sp, "es", "corpus")])
                for sp in SPLITS}
    align_q = {"development": {}}
    for qe, cs in qset["en"].items():
        mapped = {align_cv["development"][c] for c in cs}
        hits = [q for q, cs2 in qset["es"].items() if cs2 == mapped]
        if len(hits) == 1:
            align_q["development"][qe] = hits[0]
    align_q["test"] = align_jds(docs[("test", "en", "queries")], docs[("test", "es", "queries")])

    gender: dict[tuple[str, str, str], str] = {}
    for sp in SPLITS:
        d, pairs = docs[(sp, "es", "corpus")], siblings[(sp, "es")]
        for a in pairs:
            b = pairs[a][0]
            if a > b:
                continue
            f, m = gender_votes(d[a], d[b])
            if f > m:
                gender[(sp, "es", a)], gender[(sp, "es", b)] = "female", "male"
            elif m > f:
                gender[(sp, "es", a)], gender[(sp, "es", b)] = "male", "female"
            else:
                gender[(sp, "es", a)] = gender[(sp, "es", b)] = "undetermined"
        for es_id, en_id in {v: k for k, v in align_cv[sp].items()}.items():
            gender[(sp, "en", en_id)] = gender.get((sp, "es", es_id), "undetermined")

    cv_rows, job_rows, pair_rows = [], [], []
    for (sp, lg, kind), d in docs.items():
        s, other = SPLITS[sp], "es" if lg == "en" else "en"
        if kind == "corpus":
            pairs = siblings[(sp, lg)]
            amap = align_cv[sp] if lg == "en" else {v: k for k, v in align_cv[sp].items()}
            for k, t in d.items():
                sib, j = pairs[k]
                base = min(k, sib, key=int)
                g = gender.get((sp, lg, k), "undetermined")
                variant = ("" if k == base else f"gender={g};sibling_of={lg}-{s}-{base};"
                           "also_differs=name,contact,origin_linked_entities")
                cv_rows.append({
                    "cv_id": f"{lg}-{s}-{k}", "text": t.strip(), "lang": lg, "occupation": "",
                    "base_cv_id": f"{lg}-{s}-{base}", "variant": variant, "split": s,
                    "source_file": f"TaskA.zip:TaskA/{sp}/{lg}/corpus/{k}",
                    "extra": json.dumps({
                        "c_id": k, "sibling_cv_id": f"{lg}-{s}-{sib}",
                        "sibling_jaccard5": round(j, 4), "gender_inferred": g,
                        "gender_inference": GENDER_HOW, "designated_base_rule": BASE_RULE,
                        f"aligned_{other}_cv_id": (f"{other}-{s}-{amap[k]}" if k in amap
                                                   else None),
                        "dev_positive_q_ids": (sorted(pos_by_cv[lg].get(k, [])) if s == "dev"
                                               else None)}, ensure_ascii=False),
                })
        else:
            qa = align_q[sp] if lg == "en" else {v: k for k, v in align_q[sp].items()}
            for k, t in d.items():
                title, how, first = extract_title(t)
                job_rows.append({
                    "job_id": f"{lg}-{s}-q{k}", "text": t.strip(), "title": title,
                    "occupation": "", "lang": lg, "split": s,
                    "source_file": f"TaskA.zip:TaskA/{sp}/{lg}/queries/{k}",
                    "extra": json.dumps({
                        "q_id": k, "title_extraction": how, "first_line": first,
                        f"aligned_{other}_job_id": (f"{other}-{s}-q{qa[k]}" if k in qa
                                                    else None),
                        "alignment_method": JD_ALIGN_HOW,
                        "n_dev_positives": len(qset[lg].get(k, [])) if s == "dev" else None},
                        ensure_ascii=False),
                })
    for lg in LANGS:
        for i, r in enumerate(qrels[lg]):
            pair_rows.append({
                "pair_id": f"{lg}-dev-q{r[0]}-{r[2]}", "cv_id": f"{lg}-dev-{r[2]}",
                "job_id": f"{lg}-dev-q{r[0]}", "label": r[3], "label_kind": "human_relevance",
                "split": "dev",
                "extra": json.dumps({"q_id": r[0], "iter": r[1], "c_id": r[2], "relevance": r[3],
                                     "qrels_row": i, "note": QRELS_NOTE}, ensure_ascii=False),
            })
    cvs, jobs, prs = pd.DataFrame(cv_rows), pd.DataFrame(job_rows), pd.DataFrame(pair_rows)
    if not (cvs["cv_id"].is_unique and jobs["job_id"].is_unique and prs["pair_id"].is_unique):
        raise ValueError("duplicate staged ids")
    if not (set(prs["cv_id"]) <= set(cvs["cv_id"]) and set(prs["job_id"]) <= set(jobs["job_id"])):
        raise ValueError("a qrels row points at a CV or JD that is not staged")
    tables = {"cvs": cvs, "jobs": jobs, "pairs": prs}
    for name, df in tables.items():
        write_table(df, out / f"{name}.parquet")
    return {name: len(df) for name, df in tables.items()}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
