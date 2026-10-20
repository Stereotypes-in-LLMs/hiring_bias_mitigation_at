"""Stage LLMHiringFairnessSG (Singapore; 100 JDs, 4,100 CVs): raw/ -> seven parquet tables.

    python -I scripts/external/stage_llmhiringfairness_sg.py <raw dir> <staged dir>

The release pairs each JD with one neutral CV the authors generated for it and 40 demographic
variants of that CV (Chinese/Malay/Tamil/Caucasian x female/male x 5 cue sets from
demography.json). A variant is the neutral CV with an 'Additional Information' block appended
(Languages, Activities, Volunteering, Hobbies) whose content carries the identity cue. Tables:

    cvs.parquet              100 neutral bases + 4,000 variants (dataset/resumes)
    jobs.parquet             100 JDs; `text` renders the authors' LLM-structured JD, the
                             scraped and cleaned originals are in extra
    pairs.parquet            100 base pairs (neutral CV <-> its JD), label none; the 40 variant
                             ids of each set are in extra.counterfactual_variant_cv_ids
    cvs_ablated.parquet      20,500 files of dataset/ablated_resumes (cue fields removed one
                             level at a time)
    cvs_v1_templates.parquet 100 dataset/neutral_resume_templates files: an earlier prompt's
                             CVs, NOT the bases of dataset/resumes
    variant_diffs.parquet    one row per variant: word diff against its base and a class, so
                             an eval can keep only clean appends
    cue_inventory.parquet    one row per demography.json cue set with heuristic flags

Why the diff and the flags: a counterfactual is only clean if the variant differs from its base
in the cue block alone (`diff_class == 'pure_append_of_cue_block'`), and some cue sets carry
job-relevant experience (captain of a team, building PCs) or a religion/military cue that this
project audits separately. Those are measured here, at staging, so every eval reads the same
answer. The keyword lists are the acquirer's judgment calls; PROFILE.md documents them.

Reads raw/LLMHiringFairnessSG/{data/jds_structured.csv, demography.json, dataset/...}. The
'Structured JD' column holds Python dict literals and is parsed with `ast.literal_eval`, which
evaluates literals only; nothing from the repository is imported or executed.
"""

from __future__ import annotations

import ast
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import dumps, main, marker_patterns, write_table  # noqa: E402

SLUG = "llmhiringfairness_sg"
TABLES = ("cvs", "jobs", "pairs", "cvs_ablated", "cvs_v1_templates", "variant_diffs",
          "cue_inventory")
REQUIRES = ()
REPO = "LLMHiringFairnessSG"

FNAME_RE = re.compile(
    r"^job_(\d{3})_(neutral|chinese|malay|tamil|caucasian)_(?:(male|female)_)?(\d+)\.txt$")
HDR_RE = re.compile(
    r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?\*{0,2}Additional Information\*{0,2}:?[ \t]*$")
APPEND_RE = re.compile(
    r"\s*(?:#{1,6}[ \t]*)?\*{0,2}Additional Information\*{0,2}[ \t]*\n(?:\s*(?:[*-][ \t]+)?"
    r"(?:Languages|Activities|Volunteering|Hobbies): [^\n]*\n?){1,4}\s*")
FIELD_LINE = re.compile(r"^\s*(?:[*-]\s*)?(Languages|Activities|Volunteering|Hobbies)\s*:\s*(.*)$")
FIELD_ORDER = ["Languages", "Activities", "Volunteering", "Hobbies"]
DEMO_KEY = {"Languages": "languages", "Activities": "cca", "Volunteering": "volunteering",
            "Hobbies": "hobbies"}
GENERATOR = ("gemini-2.5-pro per paper (arXiv 2603.05189v3, Sec. 3.1); not stated in repo; "
             "structure matches prompts.construct_resume_prompt (v2)")
PAIRING = ("synthetic_construction: the neutral CV was LLM-generated for this JD (filename job "
           "id); the generation prompt imposes controlled weaknesses (fewer years than required, "
           "one missing secondary skill, most recent role in a related but different industry)")
TEMPLATE_NOTE = ("v1-prompt CV with [LANGUAGES]/[ACTIVITIES]/[VOLUNTEERING] placeholders; a "
                 "different CV from dataset/resumes/<job>_neutral_0 (see PROFILE.md)")

#: Broad flags for cue content a recruiter could read as experience.
FLAG_RES = {
    "leadership_title": re.compile(r"(?i)\b(captain|president|leader|lead|head of|sergeant)\b"),
    "technical_skill": re.compile(
        r"(?i)\b(hardware|pcs?|electronics|robotics|audio|visual|av|technical|sound system|"
        r"repair\w*|maintenance|design & technology|astrophysics|logistics|tentage|setup|"
        r"setting up|fixing|fermentation science)\b"),
    "organising_teaching_coaching": re.compile(
        r"(?i)\b(organis\w*|organiz\w*|running|leading|coach\w*|teaching|referee\w*|facilitator|"
        r"fundrais\w*|admin support|event planning|crew member|ushering|befriender|marshal|"
        r"crowd control)\b"),
    "commercial": re.compile(r"(?i)\b(for sale|sales|entrepreneurship|bake sale)\b"),
}
#: Stricter flags, hand-tuned after reading all 40 cue sets (still a judgment call).
STRICT_RES = {
    "leadership_title": re.compile(
        r"(?i)\b(captain|president|head of|section leader|patrol leader|hardware lead|sergeant)\b"),
    "hands_on_technical": re.compile(
        r"(?i)(hardware|electronics|custom pcs|robotics|audio[- /]visual|\bav systems\b|"
        r"sound system|motorcycle maintenance|fixing motorcycles|home improvement/repair|"
        r"design & technology)"),
    "organising_teaching_business": re.compile(
        r"(?i)\b(organis\w*|leading a|running a|coach\w*|teaching|facilitator|fundrais\w*|"
        r"admin support|event planning|entrepreneurship|for sale|bake sale|pe teacher)\b"),
}
#: Occupation -> cue content a recruiter could read as relevant to that job (heuristic).
OCC_RELEVANT = [
    (r"Software Engineer|Cybersecurity Specialist|Data Scientist",
     r"(?i)(hardware|electronics|custom pcs|robotics|audio[- /]visual|\bav systems\b|"
     r"sound system|astrophysics)"),
    (r"Civil Engineer|Process Engineer",
     r"(?i)(design & technology|robotics|hardware|electronics|tentage|home improvement|repair|"
     r"maintenance|fixing|fermentation science)"),
    (r"Preschool Teacher",
     r"(?i)\b(teaching|kids|children|young girls|primary school|face-painting|younger siblings|"
     r"coach\w*|youth)\b"),
    (r"Nurse",
     r"(?i)(red crescent|st\. john ambulance|befriender|elderly|nursing home|disabled|caring for)"),
    (r"Psychologist",
     r"(?i)(befriender|welfare|elderly|nursing home|disabled|youth programs|underprivileged|"
     r"caring for)"),
    (r"Waiter",
     r"(?i)\b(cooking|baking|cuisine|recipes|meals|food|café|barbecues|brunch|serving|pastries|"
     r"cakes)\b"),
    (r"HR Executive|Marketing Executive|Customer Service Manager",
     r"(?i)(organis|event planning|fundrais|instagram|blogging|student council|welfare|debate|"
     r"oratorical|model united nations|entrepreneurship)"),
    (r"Government Official|Paralegal|Inhouse Counsel",
     r"(?i)(debat|oratorical|model united nations|politics|student council)"),
    (r"Accountant|Accounts Clerk|Accountant Manager|Finance Analyst|Fund Manager",
     r"(?i)(chess|sudoku|playing go|strategy games|entrepreneurship|sales|fundrais|"
     r"student council)"),
]
RELIGION_RE = re.compile(
    r"(?i)\b(mosque|temple|church|islamic|prayers?|thaipusam|deepavali|arabic|methodist|anglican|"
    r"chij)\b")
MILITARY_RE = re.compile(r"(?i)\b(military|cadet|navy|army|ndp|police)\b")
STOP = set("""a an and the of for at in on to with by from or as is are be this that these those
our your their its it we you they he she i my me us will can may should must not no all any each
other into over under per via etc e g eg including include includes such well also within across
using use used work working experience ability able strong good excellent skills skill team
teams job role candidate candidates years year""".split())  # noqa: SIM905 -- a word list reads better as text


def read_txt(p: Path) -> tuple[str, bool]:
    raw = p.read_bytes()
    return raw.decode("utf-8-sig"), raw.startswith(b"\xef\xbb\xbf")


def cue_fields(lines) -> dict[str, str]:
    fields = {}
    for line in lines:
        m = FIELD_LINE.match(line.strip())
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def split_cue_block(text: str) -> tuple[str, dict[str, str], str]:
    """(body, {field: value}, block) split at the 'Additional Information' header."""
    m = HDR_RE.search(text)
    if not m:
        return text, {}, ""
    block = text[m.start():]
    return text[:m.start()], cue_fields(block.splitlines()[1:]), block


def word_diff(a: str, b: str):
    ta, tb = a.split(), b.split()
    sm = difflib.SequenceMatcher(None, ta, tb, autojunk=False)
    ins, dele, rep = [], [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            ins.append((j1, tb[j1:j2]))
        elif op == "delete":
            dele.append((i1, ta[i1:i2]))
        elif op == "replace":
            rep.append((i1, ta[i1:i2], j1, tb[j1:j2]))
    return ins, dele, rep, sm.ratio()


def render_jd(title: str, sj: dict) -> str:
    """The structured JD as plain text, sections in a fixed order, then any unknown keys."""
    lines = [f"Job Title: {title}"] if title else []
    labels = [("job_summary", "Job Summary"), ("company_description", "Company Description"),
              ("responsibilities", "Responsibilities"),
              ("required_qualifications", "Required Qualifications"),
              ("preferred_qualifications", "Preferred Qualifications"),
              ("benefits_and_perks", "Benefits and Perks")]
    known = {k for k, _ in labels}
    for k, lab in labels + [(k, k.replace("_", " ").title()) for k in sj if k not in known]:
        v = sj.get(k)
        if v in (None, "", [], {}):
            continue
        lines.append("")
        if isinstance(v, list):
            lines.append(f"{lab}:")
            lines.extend(f"- {str(x).strip()}" for x in v if str(x).strip())
        elif isinstance(v, dict):
            lines.append(f"{lab}:")
            lines.extend(f"- {kk}: {vv}" for kk, vv in v.items())
        else:
            lines.append(f"{lab}: {str(v).strip()}")
    return "\n".join(lines).strip()


def content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z&+#'-]{2,}", text.lower()) if t not in STOP}


def load_jobs(repo: Path) -> tuple[pd.DataFrame, dict[str, str], dict[str, dict]]:
    jdx = pd.read_csv(repo / "data" / "jds_structured.csv", encoding="utf-8-sig", dtype=str,
                      keep_default_na=False)
    rows, text_by_job, meta = [], {}, {}
    for i, r in jdx.iterrows():
        job_id = f"job_{i:03d}"  # positional, exactly as the authors' utils.load_jds assigns it
        sj = ast.literal_eval(r["Structured JD"]) if r["Structured JD"].startswith("{") else {}
        text = render_jd(r["Scraped Job Title"].strip() or r["Job Title"].strip(), sj)
        text_by_job[job_id] = text
        meta[job_id] = {"occupation": r["Occupation"], "industry": r["Industry"],
                        "title": r["Job Title"]}
        extra = {c: r[c] for c in ("Industry", "Occupation", "Job Title", "Platform", "Link",
                                   "SSIC 2025 2D", "SSOC 2024", "Remarks", "Scraped Job Title",
                                   "Scraped Job Description", "Cleaned JD")}
        extra.update({
            "Structured JD": sj,
            # the exact JD string the authors' evaluation code puts into its prompts
            "eval_prompt_jd_text": str(sj),
            "row_index_in_jds_structured_csv": int(i),
        })
        rows.append({"job_id": job_id, "text": text, "title": r["Job Title"].strip(),
                     "occupation": r["Occupation"].strip(), "lang": "en", "split": "",
                     "source_file": "data/jds_structured.csv", "extra": dumps(extra)})
    return pd.DataFrame(rows), text_by_job, meta


def variant_diffs(parsed, texts, base_name, demo, jd_text) -> pd.DataFrame:
    demo_lookup: dict[tuple, list] = {}
    for basket, items in demo.items():
        for idx, it in enumerate(items):
            key = tuple((it.get(DEMO_KEY[f]) or "").strip() for f in FIELD_ORDER)
            demo_lookup.setdefault(key, []).append((basket, idx))
    past_edu = {(b, i): (it.get("past_education") or "")
                for b, its in demo.items() for i, it in enumerate(its)}
    rows = []
    for name, (j, eth, gen, k) in sorted(parsed.items()):
        if eth == "neutral":
            continue
        job = f"job_{j}"
        base, var = texts[base_name[job]].strip(), texts[name].strip()
        _body, fields, _block = split_cue_block(var)
        prefix_ok = var.startswith(base)
        suffix = var[len(base):] if prefix_ok else None
        ins, dele, rep, ratio = word_diff(base, var)
        mh = HDR_RE.search(var)
        hdr_idx = len(var[:mh.start()].split()) if mh else len(var.split())
        if mh is None:
            blk_fmt = "none"
        else:
            after = var[mh.end():]
            blk_fmt = (("hash_header" if mh.group(0).strip().startswith("#") else "bold_header")
                       + ("+blank_line" if after.startswith("\n\n") else "")
                       + ("+bulleted_fields"
                          if re.match(r"\s*[*-][ \t]+(Languages|Activities)", after) else ""))
        out_ins = sum(len(w) for (j1, w) in ins if j1 < hdr_idx)
        out_del = sum(len(w) for (_, w) in dele)
        out_rep = sum(max(len(a), len(b)) for (_, a, j1, b) in rep if j1 < hdr_idx)
        in_block = (sum(len(w) for (j1, w) in ins if j1 >= hdr_idx)
                    + sum(len(b) for (_, a, j1, b) in rep if j1 >= hdr_idx))
        basket = f"{eth}_{gen}"
        matches = demo_lookup.get(tuple(fields.get(f, "").strip() for f in FIELD_ORDER), [])
        demo_entry = demo[basket][int(k)] if int(k) < len(demo.get(basket, [])) else None
        field_eq = {f: (demo_entry is not None and fields.get(f, "").strip()
                        == (demo_entry.get(DEMO_KEY[f]) or "").strip())
                    for f in FIELD_ORDER}
        pe = past_edu.get((basket, int(k)), "")
        if prefix_ok and APPEND_RE.fullmatch(suffix or ""):
            cls = "pure_append_of_cue_block"
        elif out_ins + out_del + out_rep == 0:
            cls = "no_word_change_outside_cue_block_but_not_clean_append"
        else:
            cls = "changes_outside_cue_block"
        cue_text = " ".join(fields.get(f, "") for f in ["Activities", "Volunteering", "Hobbies"])
        overlap = sorted(content_tokens(cue_text) & content_tokens(jd_text[job]))
        rows.append({
            "cv_id": name[:-4], "base_cv_id": base_name[job][:-4], "job_id": job,
            "ethnicity": eth, "gender": gen, "cue_index": int(k), "diff_class": cls,
            "cue_block_format": blk_fmt, "base_is_exact_prefix": prefix_ok,
            "words_inserted_outside_cue_block": int(out_ins),
            "words_deleted_from_base": int(out_del),
            "words_replaced_outside_cue_block": int(out_rep), "words_in_cue_block": int(in_block),
            "word_seq_ratio": float(ratio),
            "fields_present": "+".join(f for f in FIELD_ORDER if f in fields),
            "cue_matches_demography_json_entry": (basket, int(k)) in matches,
            "matched_demography_entries": json.dumps(matches),
            **{f"field_eq_{f.lower()}": bool(v) for f, v in field_eq.items()},
            "past_education_in_text": bool(pe and pe in var), "past_education": pe,
            "n_languages": len([x for x in fields.get("Languages", "").split(",") if x.strip()]),
            "cue_jd_token_overlap": json.dumps(overlap), "n_cue_jd_token_overlap": len(overlap),
            "variant_marker_patterns": json.dumps(marker_patterns(var)),
            "variant_len": len(var), "added_chars": len(var) - len(base),
        })
    return pd.DataFrame(rows)


def cue_inventory(demo: dict) -> pd.DataFrame:
    rows = []
    for basket, items in demo.items():
        if basket == "neutral":
            continue
        for idx, it in enumerate(items):
            row = {"basket": basket, "cue_index": idx,
                   "past_education_unused": it.get("past_education"),
                   "languages": it.get("languages"), "activities": it.get("cca"),
                   "volunteering": it.get("volunteering"), "hobbies": it.get("hobbies"),
                   "n_languages": len([x for x in (it.get("languages") or "").split(",")
                                       if x.strip()])}
            txt = " ".join(str(it.get(x) or "") for x in ["cca", "volunteering", "hobbies"])
            with_langs = txt + " " + (it.get("languages") or "")
            for fl, rx in FLAG_RES.items():
                row[fl] = sorted({m.group(0) for m in rx.finditer(txt)})
            row["religion_linked"] = sorted({m.group(0) for m in RELIGION_RE.finditer(with_langs)})
            row["military_or_uniformed_linked"] = sorted(
                {m.group(0) for m in MILITARY_RE.finditer(txt)})
            row["project_marker_hits"] = marker_patterns(with_langs)
            for fl, rx in STRICT_RES.items():
                row[f"strict_{fl}"] = sorted({m.group(0) for m in rx.finditer(txt)})
            row["experience_bearing_strict"] = bool(
                row["strict_leadership_title"] or row["strict_hands_on_technical"]
                or row["strict_organising_teaching_business"])
            row["experience_bearing_heuristic"] = bool(
                row["leadership_title"] or row["technical_skill"]
                or row["organising_teaching_coaching"] or row["commercial"])
            rows.append(row)
    return pd.DataFrame(rows)


def occupation_relevance(dd: pd.DataFrame, cue_df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    eb = {(r.basket, r.cue_index): r.experience_bearing_strict for r in cue_df.itertuples()}
    cue_txt = {(r.basket, r.cue_index): " ".join(str(x or "") for x in
                                                  [r.activities, r.volunteering, r.hobbies])
               for r in cue_df.itertuples()}
    rows = []
    for r in dd.itertuples():
        occ = meta[r.job_id]["occupation"]
        key = (f"{r.ethnicity}_{r.gender}", r.cue_index)
        rel = None
        for occ_re, cue_re in OCC_RELEVANT:
            if re.fullmatch(occ_re, occ):
                rel = sorted({m if isinstance(m, str) else m[0]
                              for m in re.findall(cue_re, cue_txt[key])})
                break
        rows.append({"cv_id": r.cv_id, "occupation": occ,
                     "experience_bearing_strict": bool(eb[key]),
                     "occupation_relevant_cue_terms": json.dumps(rel or []),
                     "occupation_relevant": bool(rel)})
    return pd.DataFrame(rows)


def ablated_rows(adir: Path, texts: dict, base_name: dict, meta: dict) -> list[dict]:
    out = []
    for lev in sorted(p.name for p in adir.iterdir() if p.is_dir()):
        for p in sorted((adir / lev).glob("*.txt")):
            m = FNAME_RE.match(p.name)
            if not m:
                continue
            j, eth, gen, k = m.groups()
            t, bom = read_txt(p)
            t = t.strip()
            ref = texts.get(p.name, "").strip()
            base = texts[base_name[f"job_{j}"]].strip()
            # cue fields are whatever follows the base (an ablation may remove the header too)
            fields = cue_fields((t[len(base):] if t.startswith(base) else t).splitlines())
            fp = "+".join(f for f in FIELD_ORDER if f in fields)
            job = f"job_{j}"
            variant = (f"ablation={lev}" if eth == "neutral" else
                       f"ethnicity={eth};gender={gen};cue={int(k)};ablation={lev};"
                       f"fields={fp or 'none'}")
            out.append({"cv_id": f"ablated/{lev}/{p.name[:-4]}", "text": t, "lang": "en",
                        "occupation": meta[job]["occupation"], "base_cv_id": f"job_{j}_neutral_0",
                        "variant": variant, "split": "",
                        "source_file": f"dataset/ablated_resumes/{lev}/{p.name}",
                        "extra": dumps({"job_id": job, "ablation_level": lev, "fields_present": fp,
                                        "identical_to_dataset_resumes_file": t == ref,
                                        "identical_to_neutral_base": t == base,
                                        "had_utf8_bom": bom})})
    return out


def stage(raw: Path, out: Path) -> dict[str, int]:
    repo = raw / REPO
    jobs, jd_text, meta = load_jobs(repo)
    demo = json.loads((repo / "demography.json").read_text(encoding="utf-8"))

    rdir = repo / "dataset" / "resumes"
    parsed = {p.name: FNAME_RE.match(p.name).groups()
              for p in sorted(rdir.glob("*")) if FNAME_RE.match(p.name)}
    per_job: dict[str, Counter] = defaultdict(Counter)
    for j, eth, gen, _k in parsed.values():
        per_job[f"job_{j}"][f"{eth}_{gen}" if gen else eth] += 1
    texts, boms = {}, {}
    for name in parsed:
        texts[name], boms[name] = read_txt(rdir / name)
    base_name = {f"job_{j}": f"job_{j}_neutral_0.txt"
                 for (j, e, _g, _k) in parsed.values() if e == "neutral"}

    dd = variant_diffs(parsed, texts, base_name, demo, jd_text)
    cue_df = cue_inventory(demo)
    dd = dd.merge(occupation_relevance(dd, cue_df, meta), on="cv_id", how="left")
    cue_store = cue_df.copy()
    for c in (list(FLAG_RES) + [f"strict_{f}" for f in STRICT_RES]
              + ["religion_linked", "military_or_uniformed_linked", "project_marker_hits"]):
        cue_store[c] = cue_store[c].map(json.dumps)

    dd_idx = dd.set_index("cv_id")
    cv_rows = []
    for name, (j, eth, gen, k) in sorted(parsed.items()):
        job, cv_id = f"job_{j}", name[:-4]
        t = texts[name].strip()
        _body, fields, _block = split_cue_block(t)
        extra = {"job_id": job, "source_filename": name, "had_utf8_bom": bool(boms[name]),
                 "industry": meta[job]["industry"], "generated_for_job_title": meta[job]["title"],
                 "synthetic": True, "generator": GENERATOR}
        if eth == "neutral":
            variant = ""
            extra.update({"role": "neutral_baseline",
                          "counterfactual_set_size": int(sum(per_job[job].values()))})
        else:
            variant = f"ethnicity={eth};gender={gen};cue={int(k)}"
            r = dd_idx.loc[cv_id]
            extra.update({
                "role": "demographic_variant", "ethnicity_source_label": eth,
                "ethnicity_readme_label": {"tamil": "Indian"}.get(eth, eth.capitalize()),
                "gender": gen, "cue_index_in_demography_json": int(k), "cue_fields": fields,
                "cue_matches_demography_json_entry": bool(r["cue_matches_demography_json_entry"]),
                "diff_class": r["diff_class"],
                "words_changed_outside_cue_block": int(
                    r["words_inserted_outside_cue_block"] + r["words_deleted_from_base"]
                    + r["words_replaced_outside_cue_block"]),
            })
        cv_rows.append({"cv_id": cv_id, "text": t, "lang": "en",
                        "occupation": meta[job]["occupation"], "base_cv_id": f"job_{j}_neutral_0",
                        "variant": variant, "split": "", "source_file": f"dataset/resumes/{name}",
                        "extra": dumps(extra)})

    pair_rows = []
    for job in sorted(base_name):
        var_ids = sorted(n[:-4] for n, g in parsed.items()
                         if f"job_{g[0]}" == job and g[1] != "neutral")
        pair_rows.append({
            "pair_id": f"{job}__{base_name[job][:-4]}", "cv_id": base_name[job][:-4],
            "job_id": job, "label": None, "label_kind": "none", "split": "",
            "extra": json.dumps({
                "pairing": PAIRING, "occupation": meta[job]["occupation"],
                "industry": meta[job]["industry"], "counterfactual_variant_cv_ids": var_ids,
                "n_variants": len(var_ids),
                "original_tasks": ["pairwise variant-vs-neutral A/B/Tie (both orders)",
                                   "1-100 score; top-score shortlist per job"],
            }),
        })
    pairs = pd.DataFrame(pair_rows)
    pairs["label"] = pairs["label"].astype("string")

    templates = []
    for p in sorted((repo / "dataset" / "neutral_resume_templates").glob("*.txt")):
        t, bom = read_txt(p)
        job = p.stem
        templates.append({"cv_id": f"template/{job}", "text": t.strip(), "lang": "en",
                          "occupation": meta[job]["occupation"], "base_cv_id": f"template/{job}",
                          "variant": "", "split": "",
                          "source_file": f"dataset/neutral_resume_templates/{job}.txt",
                          "extra": dumps({"job_id": job, "had_utf8_bom": bom,
                                          "note": TEMPLATE_NOTE})})

    tables = {
        "cvs": pd.DataFrame(cv_rows), "jobs": jobs, "pairs": pairs,
        "cvs_ablated": pd.DataFrame(ablated_rows(repo / "dataset" / "ablated_resumes", texts,
                                                 base_name, meta)),
        "cvs_v1_templates": pd.DataFrame(templates), "variant_diffs": dd,
        "cue_inventory": cue_store,
    }
    for name, df in tables.items():
        write_table(df, out / f"{name}.parquet")
    return {name: len(df) for name, df in tables.items()}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
