"""Stage FAIRE (Fairness Assessment In Resume Evaluation): raw/ -> cvs, jobs, pairs parquet.

    data-tools-venv/bin/python -I scripts/external/stage_faire.py <raw dir> <staged dir> \
        [--kaggle-raw <kaggle_resume raw dir>]

FAIRE ships one spreadsheet, raw/FAIRE/Resumes.xlsx, sheet 'Resume'. Excel rows 4..101 are
the 98 complete sets: column A the category, B the JD, C the base CV ('Original'), then an
'Experience' and a 'Name' variant for each of seven groups (D..Q: Black, Native
American/Hawaiian, Asian, Hispanic, White, Male, Female). Rows below the block hold only
column-B paste residue; two of them (1722, 1723) are whole JDs and are staged as orphans.

Each variant is diffed against its own base at the token level and classed, because the
spreadsheet does not say what changed and the classes differ in what a decision flip means:

* name variants: `identity_swap_name_only` (a name inserted at the start, nothing else),
  `..._plus_format`, or `other`;
* experience variants: `experience_bearing` (adds non-identity words, i.e. content a
  recruiter could weigh), `identity_terms_only`, `deletion_only`, `no_content_change`, or
  `derived_from_other_cv` when under 30% of the variant's tokens come from its base (row 19's
  experience column was built from another CV).

`extra.kaggle_match` ties each base to its Kaggle Resume Dataset ID (exact after whitespace
collapse, else the best rapidfuzz ratio within the same category). That is the overlap the
external holdout exists for: every FAIRE base is a kaggle_resume CV under another id. So
staging needs kaggle_resume's raw Resume.csv, and refuses to run without it rather than
writing a file whose `kaggle_match` is silently empty.

The verifier's fix (`_scripts/faire/verify/fix01_extra_nan_to_null.py`) is built in: name
variants have no `share_base_tokens_kept` / `lossy_del_ge20pct`, and they are written as
JSON null, not the bare NaN the acquirer's pandas row produced (invalid strict JSON).
"""

from __future__ import annotations

import difflib
import io
import json
import re
import sys
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import dumps, main, require, write_table  # noqa: E402

SLUG = "faire"
TABLES = ("cvs", "jobs", "pairs")
REQUIRES = ("openpyxl", "rapidfuzz")
XLSX = "FAIRE/Resumes.xlsx"
ORPHAN_JD_ROWS = (1722, 1723)

#: Spreadsheet column index -> (axis, group, variant type, Excel letter).
COLS = {
    3: ("race", "black", "experience", "D"), 4: ("race", "black", "name", "E"),
    5: ("race", "native_american_hawaiian", "experience", "F"),
    6: ("race", "native_american_hawaiian", "name", "G"),
    7: ("race", "asian", "experience", "H"), 8: ("race", "asian", "name", "I"),
    9: ("race", "hispanic", "experience", "J"), 10: ("race", "hispanic", "name", "K"),
    11: ("race", "white", "experience", "L"), 12: ("race", "white", "name", "M"),
    13: ("gender", "male", "experience", "N"), 14: ("gender", "male", "name", "O"),
    15: ("gender", "female", "experience", "P"), 16: ("gender", "female", "name", "Q"),
}
#: Identity words: inserting only these is an identity cue, not added experience.
IDLEX = set("""black african african-american afro native indigenous tribal tribe tribes hawaiian
hawaiians pacific islander islanders asian asian-american chinese korean japanese vietnamese
filipino hispanic latino latina latinx latin latinos latinas mexican chicano white caucasian
european male female males females man men woman women girls boys girl boy he she his her him
hers himself herself mr mrs ms minority minorities ethnic american americans""".split())  # noqa: SIM905 -- a word list reads better as text
STOP = {"a", "an", "the", "of", "for", "in", "and", "to", "with", "within", "from", "on", "by",
        "at", "as", "all", "or", "its", "their", "our", "who", "that", "which", "s"}


def norm_tokens(s: str) -> list[str]:
    """Word and punctuation tokens; hyphens are their own token so 'full-time' vs 'full time'
    is a format-only change."""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[‐-―−－]", "-", s)
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.findall(r"\w+(?:['/.&]\w+)*|[^\w\s]", s)


def is_content(t: str) -> bool:
    return any(ch.isalnum() for ch in t)


def diff(base: str, var: str):
    a = [t.casefold() for t in norm_tokens(base)]
    b_orig = norm_tokens(var)
    b = [t.casefold() for t in b_orig]
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        old, new = a[i1:i2], b[j1:j2]
        ops.append({"tag": tag, "i1": i1, "content": any(is_content(t) for t in old + new),
                    "base_c": [t for t in old if is_content(t)],
                    "var_c": [t for t in new if is_content(t)]})
    return sm.ratio(), ops, a, b


def classify(rows: list[dict]) -> dict[tuple[int, int], dict]:
    """(excel row, column index) -> the diff summary that goes into the variant's extra."""
    base_tok = {x["excel_row"]: [t.casefold() for t in norm_tokens(x["base"])] for x in rows}
    # first names of the gender-name variants, the lexicon for guessing a race-name's gender
    male_first = {x["variants"][14].split()[0] for x in rows}
    female_first = {x["variants"][16].split()[0] for x in rows}
    out = {}
    for x in rows:
        base, er = x["base"], x["excel_row"]
        bcont = sum(1 for t in base_tok[er] if is_content(t))
        for c, (_axis, _group, vtype, _letter) in COLS.items():
            var = x["variants"][c]
            ratio, ops, a, b = diff(base, var)
            cops = [o for o in ops if o["content"]]
            ins_c = [t for o in cops for t in o["var_c"]]
            del_c = [t for o in cops for t in o["base_c"]]
            new_words = [t for t in ins_c if t not in IDLEX and t not in STOP]
            rec = {"ratio_vs_own_base": round(ratio, 4), "ins_content_tokens": len(ins_c),
                   "del_content_tokens": len(del_c), "new_non_identity_words": len(new_words),
                   "first_name_gender_by_lexicon": None, "best_other_base": None,
                   "share_base_tokens_kept": None, "lossy_del_ge20pct": None, "name": ""}
            if vtype == "name":
                ok = len(cops) == 1 and cops[0]["tag"] == "insert" and cops[0]["i1"] == 0
                name = " ".join(var.split()[:2]) if ok else ""
                fn = name.split()[0] if name else ""
                rec["name"] = name
                rec["first_name_gender_by_lexicon"] = (
                    "male" if fn in male_first else "female" if fn in female_first else "unknown")
                punct_only_ops = len(ops) - len(cops)
                cls = ("identity_swap_name_only" if ok and punct_only_ops == 0
                       else "identity_swap_name_plus_format" if ok else "other")
            else:
                sm2 = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
                matched = sum(m.size for m in sm2.get_matching_blocks())
                found_in_base = matched / max(len(b), 1)
                rec["share_base_tokens_kept"] = round(matched / max(len(a), 1), 4)
                rec["lossy_del_ge20pct"] = (len(del_c) / max(bcont, 1)) >= 0.20
                if ratio < 0.5:
                    best = max((difflib.SequenceMatcher(a=base_tok[k], b=b, autojunk=False)
                                .ratio(), k) for k in base_tok if k != er)
                    rec["best_other_base"] = {"excel_row": best[1], "ratio": round(best[0], 4)}
                if round(found_in_base, 4) < 0.3:
                    cls = "derived_from_other_cv"
                elif not cops:
                    cls = "no_content_change"
                elif not ins_c:
                    cls = "deletion_only"
                elif not new_words:
                    cls = "identity_terms_only"
                else:
                    cls = "experience_bearing"
            rec["class"] = cls
            out[(er, c)] = rec
    return out


def read_kaggle(kaggle_raw: Path) -> pd.DataFrame:
    csv_path = kaggle_raw / "extracted" / "Resume" / "Resume.csv"
    zip_path = kaggle_raw / "resume-dataset.zip"
    if csv_path.exists():
        return pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            return pd.read_csv(io.BytesIO(z.read("Resume/Resume.csv")), dtype=str,
                               keep_default_na=False)
    raise FileNotFoundError(
        f"FAIRE staging cross-matches its base CVs to kaggle_resume, and neither {csv_path} nor "
        f"{zip_path} exists. Fetch kaggle_resume first "
        "(scripts/fetch_external_data.py --source kaggle_resume).")


def kaggle_matches(rows: list[dict], kaggle: pd.DataFrame) -> dict[int, dict]:
    from rapidfuzz import fuzz, process

    norm = kaggle["Resume_str"].map(lambda s: " ".join(s.split()))
    exact: dict[str, list[tuple[str, str]]] = {}
    for n, kid, cat in zip(norm, kaggle["ID"], kaggle["Category"]):
        exact.setdefault(n, []).append((kid, cat))
    out = {}
    for x in rows:
        nb = " ".join(x["base"].split())
        if nb in exact:
            hits = exact[nb]
            out[x["excel_row"]] = {"kaggle_ids": [h[0] for h in hits],
                                   "kaggle_category": [h[1] for h in hits],
                                   "match": "exact_ws_normalised"}
            continue
        same_cat = kaggle["Category"] == x["category"]
        m = process.extractOne(nb, norm[same_cat].tolist(), scorer=fuzz.ratio)
        if m:
            out[x["excel_row"]] = {"kaggle_ids": [kaggle.loc[same_cat, "ID"].iloc[m[2]]],
                                   "kaggle_category": [x["category"]],
                                   "match": f"fuzzy_ratio={m[1]:.1f}"}
    return out


def first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def stage(raw: Path, out: Path, kaggle_raw: Path | None = None) -> dict[str, int]:
    require(*REQUIRES)
    kaggle = read_kaggle(kaggle_raw or raw.parent.parent / "kaggle_resume" / "raw")
    sheet = pd.read_excel(raw / XLSX, sheet_name="Resume", header=None)
    data = sheet.iloc[3:101].reset_index(drop=True)
    if len(data) != 98 or not data.notna().all().all():
        raise ValueError("expected 98 complete rows at Excel rows 4..101 of sheet 'Resume'")
    rows = [{"excel_row": i + 4, "category": str(r[0]).strip(), "jd": str(r[1]),
             "base": str(r[2]), "variants": {c: str(r[c]) for c in COLS}}
            for i, r in data.iterrows()]
    summaries = classify(rows)
    kmatch = kaggle_matches(rows, kaggle)

    # two base rows carry the same CV text; each points at the other
    dup_base, seen = {}, {}
    for x in rows:
        nb = " ".join(x["base"].split())
        if nb in seen:
            dup_base[x["excel_row"]] = seen[nb]
            dup_base[seen[nb]] = x["excel_row"]
        else:
            seen[nb] = x["excel_row"]

    cv_rows, job_rows, pair_rows = [], [], []
    for x in rows:
        er = x["excel_row"]
        bid, jid = f"faire_r{er:03d}", f"faire_r{er:03d}_jd"
        cv_rows.append({
            "cv_id": bid, "text": x["base"].strip(), "lang": "en", "occupation": x["category"],
            "base_cv_id": bid, "variant": "", "split": "",
            "source_file": f"Resumes.xlsx#Resume!C{er}",
            "extra": dumps({"excel_row": er, "column": "C", "header": "Original",
                            "category": x["category"], "job_id": jid,
                            "kaggle_match": kmatch.get(er, {}),
                            "duplicate_base_text_of_row": dup_base.get(er),
                            "faire_native_id": None}),
        })
        jd = x["jd"].strip()
        job_rows.append({"job_id": jid, "text": jd, "title": first_line(jd),
                         "occupation": x["category"], "lang": "en", "split": "",
                         "source_file": f"Resumes.xlsx#Resume!B{er}",
                         "extra": json.dumps({"excel_row": er, "column": "B",
                                              "paired_cv_id": bid})})
        pair_rows.append({"pair_id": bid, "cv_id": bid, "job_id": jid, "label": None,
                          "label_kind": "none", "split": "",
                          "extra": json.dumps({"excel_row": er, "category": x["category"]})})
        for c, (axis, group, vtype, letter) in COLS.items():
            s = summaries[(er, c)]
            variant = (f"axis={axis};name={s['name'] if vtype == 'name' else ''};"
                       f"gender={group if axis == 'gender' else ''};"
                       f"ethnicity={group if axis == 'race' else ''};type={vtype}")
            cv_rows.append({
                "cv_id": f"{bid}_{axis}_{group}_{vtype}", "text": x["variants"][c].strip(),
                "lang": "en", "occupation": x["category"], "base_cv_id": bid,
                "variant": variant, "split": "", "source_file": f"Resumes.xlsx#Resume!{letter}{er}",
                "extra": dumps({
                    "excel_row": er, "column": letter, "header_group": group,
                    "header_type": vtype.capitalize(), "job_id": jid, "diff_class": s["class"],
                    "ratio_vs_own_base": s["ratio_vs_own_base"],
                    "ins_content_tokens": s["ins_content_tokens"],
                    "del_content_tokens": s["del_content_tokens"],
                    "new_non_identity_words": s["new_non_identity_words"],
                    "first_name_gender_by_lexicon": s["first_name_gender_by_lexicon"],
                    "best_other_base": s["best_other_base"],
                    "share_base_tokens_kept": s["share_base_tokens_kept"],
                    "lossy_del_ge20pct": s["lossy_del_ge20pct"]}),
            })
    for er in ORPHAN_JD_ROWS:
        jd = str(sheet.iat[er - 1, 1]).strip()
        job_rows.append({"job_id": f"faire_r{er}_jd", "text": jd, "title": first_line(jd),
                         "occupation": "", "lang": "en", "split": "orphan_no_cv",
                         "source_file": f"Resumes.xlsx#Resume!B{er}",
                         "extra": json.dumps({"excel_row": er, "note": "stray JD below the data "
                                              "block, no CV/variants, no category"})})
    pairs = pd.DataFrame(pair_rows)
    # an all-null label column would be written with Arrow type null; keep it a string column
    pairs["label"] = pairs["label"].astype("string")
    tables = {"cvs": pd.DataFrame(cv_rows), "jobs": pd.DataFrame(job_rows), "pairs": pairs}
    for name, df in tables.items():
        write_table(df, out / f"{name}.parquet")
    return {name: len(df) for name, df in tables.items()}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0],
         options=(("--kaggle-raw", "kaggle_resume's raw/ directory (default: the sibling "
                                   "<root>/kaggle_resume/raw)"),))
