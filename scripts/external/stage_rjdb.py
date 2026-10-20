"""Stage RJDB (Megagon Labs): raw/ -> cvs.parquet, jobs.parquet, pairs.parquet.

    data-tools-venv/bin/python -I scripts/external/stage_rjdb.py <raw dir> <staged dir>

Each RJDB record is a triple: a GPT-4 job description, a CV generated from the same
skill/experience seed ("Resume-matched"), and a GPT-4 rewrite of that CV with skills removed
and/or the last experience shortened or dropped ("Resume-unmatched"). The matched CV is the
base; the unmatched one is a weakened variant that points at it, and both are paired with the
JD (`match` / `mismatch`), which is what makes RJDB a qualification-sensitivity control.

`variant` summarises what the rewrite was told to change, parsed from the record's
`Filtered-information`. The acquirer's parser lacked `re.DOTALL`, so the 440 records whose
`Filtered-information.Experience` spans several lines got `exp=shortened` with no year ranges
or `exp=dropped:` with no target, and empty `experience_change_from/to`. The verifier
re-parsed exactly those rows (`_scripts/rjdb/verify/v11_fix_multiline_variants.py`); the
parser here is the fixed one, so a re-stage gives the verified file. A dropped target that
spans lines has its whitespace collapsed so the variant string stays on one line; a one-line
target is kept verbatim, as it always was.

Occupation is the seed experience that best matches the JD's parsed title (rapidfuzz
token-set ratio), because the seed's first experience is often not the role the JD is for.
Reads raw/gdrive/train.jsonl, raw/repo/data/dev.jsonl and raw/repo/data/test.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import dumps, main, marker_indices, require, write_table  # noqa: E402

SLUG = "rjdb"
TABLES = ("cvs", "jobs", "pairs")
REQUIRES = ("rapidfuzz",)
SPLITS = (("train", "gdrive/train.jsonl"), ("dev", "repo/data/dev.jsonl"),
          ("test", "repo/data/test.jsonl"))

YEAR_RANGE = re.compile(
    r"((?:19|20)\d{2})\s*(?:-|–|—|to)\s*((?:19|20)\d{2}|present|current)", re.I)
TITLE_LINE = re.compile(r"(?i)^job[\s\-_]*title\b\s*[:\-–—]?\s*(.*)$")
NAME_RE = re.compile(
    r"(?im)^[\s\-*•]*\**\s*(?:full\s+)?name\s*\**\s*[:\-]\s*\**\s*([^\n*]+?)\s*\**\s*$")
PAIR_SEMANTICS = (
    "match = Resume-matched generated from same skill/experience seed as JD; mismatch = "
    "Resume-unmatched, GPT-4 rewrite of matched with skills removed and/or last experience "
    "shortened/dropped (weaker match, not a real hiring outcome)")


def norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def text_sha1(t: str) -> str:
    """sha1 of the whitespace-collapsed text: equal for copies that differ only in layout."""
    return hashlib.sha1(norm_ws(t).encode("utf-8")).hexdigest()


def clean_title(t: str) -> str:
    t = t.strip().strip("#*").strip()
    t = re.sub(r"^\*+|\*+$", "", t).strip()
    return t.strip("\"'“”:").strip()


def parse_title(jd: str) -> str:
    """The JD's 'Job Title' line, or the first non-empty line under a bare 'Job Title' header."""
    lines = jd.splitlines()
    for i, line in enumerate(lines):
        s = line.strip().lstrip("#").strip().strip("*").strip()
        m = TITLE_LINE.match(s)
        if m:
            rest = clean_title(m.group(1))
            if rest:
                return rest
            for nxt in lines[i + 1:]:
                t = clean_title(nxt)
                if t:
                    return t
            return ""
    return ""


def norm_occ(t: str) -> str:
    t = t.lower().replace(" - ", ", ")
    t = re.sub(r"[^a-z0-9+#./ ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def canonical_occupation(title: str, exps: list[str]) -> tuple[str, str, float]:
    """(occupation, how it was chosen, match score) from the parsed title and the seed list."""
    from rapidfuzz import fuzz

    if not exps:
        return "", "none", 0.0
    if len(exps) == 1:
        score = 100.0 if not title else float(
            fuzz.token_set_ratio(norm_occ(title), norm_occ(exps[0])))
        return exps[0], "experiences_single", score
    if not title:
        return exps[0], "experiences[0]_no_title", 0.0
    scores = [fuzz.token_set_ratio(norm_occ(title), norm_occ(e)) for e in exps]
    j = int(np.argmax(scores))
    return exps[j], "best_match_title_vs_experiences", float(scores[j])


def is_bls_like(occ: str) -> bool:
    """BLS/SOC titles are sentence case with a plural head noun ('Designers, all other');
    DICE titles are job-posting titles ('Java Developer'). A heuristic, recorded in extra."""
    if not occ or not occ[0].isupper():
        return False
    if any(c.isupper() for c in occ[1:]):
        return False
    if "all other" in occ:
        return True
    plural_like = {"police", "personnel", "staff", "clergy"}
    for seg in occ.split(","):
        words = seg.strip().split()
        if words and (words[-1].endswith("s") or words[-1] in plural_like):
            return True
    return False


def experience_change(fi: dict) -> tuple[str, str, str, str]:
    """(kind, old, new, variant fragment) parsed from Filtered-information.Experience.

    `re.S` is the verifier's fix: 440 Experience values span several lines, and without it
    the quoted spans never matched.
    """
    exp_mod = (fi.get("Experience") or "").strip()
    if exp_mod.startswith("instead of"):
        m = re.match(r'^instead of "(.*)" use "(.*)"$', exp_mod, re.S)
        old, new = (m.group(1), m.group(2)) if m else ("", "")
        ro, rn = YEAR_RANGE.search(old), YEAR_RANGE.search(new)
        desc = (f"shortened:{ro.group(1)}-{ro.group(2)}->{rn.group(1)}-{rn.group(2)}"
                if ro and rn else "shortened")
        return "shortened", old, new, desc
    if exp_mod.startswith("without including"):
        m = re.match(r'^without including "(.*)"$', exp_mod, re.S)
        old = m.group(1) if m else ""
        target = " ".join(old.split()) if "\n" in old else old
        return "dropped", old, "", f"dropped:{target}"
    if exp_mod == "the same job experience as before":
        return "unchanged", "", "", "unchanged"
    return "other", "", "", "other"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stage(raw: Path, out: Path) -> dict[str, int]:
    require(*REQUIRES)
    cv_rows, job_rows, pair_rows = [], [], []
    for split, rel in SPLITS:
        src = f"raw/{rel}"
        for idx, r in enumerate(load_jsonl(raw / rel)):
            rid = f"rjdb-{split}-{idx:05d}"
            jd = r["Job-Description"].strip()
            rm = r["Resume-matched"].strip()
            ru = r["Resume-unmatched"].strip()
            exps = list(r["Experiences"])
            title = parse_title(jd)
            occ, occ_src, occ_score = canonical_occupation(title, exps)
            fi = r["Filtered-information"]
            sk = fi.get("Skills")
            removed = list(sk.keys()) if isinstance(sk, dict) else list(sk or [])
            kind, old, new, exp_desc = experience_change(fi)
            variant = (f"filtered:skills_removed={'|'.join(removed) if removed else 'none'};"
                       f"exp={exp_desc}")
            base_extra = {
                "record_index": idx, "Skills": r["Skills"], "Experiences": exps,
                "Experiences-years": r["Experiences-years"], "text_sha1": text_sha1(rm),
            }
            name_m, name_u = NAME_RE.search(rm), NAME_RE.search(ru)
            cv_rows.append({
                "cv_id": f"{rid}-m", "text": rm, "lang": "en", "occupation": occ,
                "base_cv_id": f"{rid}-m", "variant": "", "split": split, "source_file": src,
                "extra": dumps({**base_extra, "role": "Resume-matched",
                                "name": name_m.group(1) if name_m else None,
                                "protected_marker_hits": marker_indices(rm)}),
            })
            cv_rows.append({
                "cv_id": f"{rid}-u", "text": ru, "lang": "en", "occupation": occ,
                "base_cv_id": f"{rid}-m", "variant": variant, "split": split, "source_file": src,
                "extra": dumps({"record_index": idx, "role": "Resume-unmatched",
                                "Filtered-information": fi, "removed_skills": removed,
                                "experience_change_kind": kind, "experience_change_from": old,
                                "experience_change_to": new,
                                "name": name_u.group(1) if name_u else None,
                                "text_sha1": text_sha1(ru),
                                "protected_marker_hits": marker_indices(ru)}),
            })
            job_extra = {k: v for k, v in base_extra.items() if k != "text_sha1"}
            job_rows.append({
                "job_id": f"{rid}-jd", "text": jd, "title": title, "occupation": occ,
                "lang": "en", "split": split, "source_file": src,
                "extra": dumps({**job_extra, "text_sha1": text_sha1(jd),
                                "occupation_source": occ_src, "occupation_match_score": occ_score,
                                "occupation_bls_like": is_bls_like(occ)}),
            })
            for label, cv in (("match", f"{rid}-m"), ("mismatch", f"{rid}-u")):
                pair_rows.append({
                    "pair_id": f"{rid}-{label}", "cv_id": cv, "job_id": f"{rid}-jd",
                    "label": label, "label_kind": "synthetic_construction", "split": split,
                    "extra": dumps({"record_index": idx, "triple_id": rid,
                                    "n_removed_skills": len(removed),
                                    "experience_change_kind": kind,
                                    "semantics": PAIR_SEMANTICS}),
                })
    tables = {"cvs": pd.DataFrame(cv_rows), "jobs": pd.DataFrame(job_rows),
              "pairs": pd.DataFrame(pair_rows)}
    for name, df in tables.items():
        write_table(df, out / f"{name}.parquet")
    return {name: len(df) for name, df in tables.items()}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
