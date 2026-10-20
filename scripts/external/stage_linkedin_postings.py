"""Stage LinkedIn Job Postings 2023-2024 (Kaggle arshkon, v13): raw/ -> jobs.parquet.

    python -I scripts/external/stage_linkedin_postings.py <raw dir> <staged dir>

One row per posting with a non-empty description (7 postings have none and are dropped).
`text` is the description stripped; `occupation` is the posting's LinkedIn job-function list
joined by ';' (coarse, sometimes 'Other', empty for ~1.7k rows). Every other postings.csv
column is kept in `extra`, with the posting's industries, job functions, benefits and company
industries joined in from the side tables.

Three things in `extra` are for pool building, not facts about the posting:

* `desc_dup_group` keys exact duplicates after whitespace collapse and casefold (multi-location
  reposts); near duplicates are not grouped here.
* `kaggle_resume_cats_by_title/industry/function` map the posting onto the 24 kaggle_resume
  Categories, so CVs and JDs from the two Kaggle sources can be paired by occupation. The title
  map is regex on `title` and is the one to trust; the industry and function maps are coarse.
* `lang` is a stopword-share heuristic: 'en' at >= 5% English stopwords, 'und' below.

Reads raw/extracted/postings.csv, jobs/{job_industries,job_skills,benefits}.csv,
mappings/{industries,skills}.csv and companies/company_industries.csv.
"""

from __future__ import annotations

import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import MARKERS_EN, dumps, main, write_table  # noqa: E402

SLUG = "linkedin_postings"
TABLES = ("jobs",)
REQUIRES = ()

#: One regex for "any marker": every pattern starts with (?i), which is stripped and re-applied.
MARKER_ANY = re.compile("|".join(f"(?:{p[4:]})" for p in MARKERS_EN), re.I)
STOP = {"the", "and", "of", "to", "in", "a", "for", "with", "on", "as", "by", "at", "an", "or",
        "from", "is", "are", "will", "you", "our", "we", "this", "that", "be", "your"}

# kaggle_resume Category -> LinkedIn fields. A posting may match several categories.
# TITLE_MAP: case-insensitive regex on the posting title (v2, after the acquirer's spot checks).
TITLE_MAP = {
    "HR": r"\bhr\b|human resources|recruit|talent acquisition|people (?:operations|partner)|"
          r"\bhrbp\b|benefits (?:specialist|administrator|analyst)",
    "DESIGNER": r"designer",
    "INFORMATION-TECHNOLOGY": r"software|developer|\bit\b|information technology|"
                              r"system(?:s)? administrator|network (?:engineer|administrator|"
                              r"technician)|devops|help ?desk|desktop support|cyber ?security|"
                              r"programmer|database|\bcloud\b|data engineer",
    "TEACHER": r"teacher|tutor\b|instructor|educator|professor|\bteaching\b|paraprofessional",
    # kaggle_resume ADVOCATE headlines are patient/customer/victim/family advocates, not lawyers
    "ADVOCATE": r"\badvocate",
    "BUSINESS-DEVELOPMENT": r"business development|\bbdr\b|partnerships? (?:manager|director)",
    "HEALTHCARE": r"nurse|\brn\b|\blpn\b|\bcna\b|physician|medical|clinical|health ?care|pharmac|"
                  r"therapist|dental|caregiver|patient",
    "FITNESS": r"fitness|personal trainer|yoga|pilates|\bgym\b|strength (?:and|&) conditioning",
    # \branch\b was dropped: it matched place names like "Highlands Ranch"
    "AGRICULTURE": r"agricultur|\bfarm(?:er|ing|worker|hand)?\b|agronom|\bcrops?\b|livestock|"
                   r"horticult|greenhouse|\bagri",
    "BPO": r"call center|contact center|customer service|customer support|customer care|\bbpo\b|"
           r"telesales",
    "SALES": r"\bsales\b|account executive|retail associate",
    "CONSULTANT": r"consultant|consulting",
    "DIGITAL-MEDIA": r"digital media|social media|digital marketing|content (?:creator|producer|"
                     r"strategist|manager)|video (?:editor|producer)|multimedia|\bseo\b",
    # \bmechanics?\b, not 'mechanic', which matched "Mechanical Engineer" and "Biomechanics"
    "AUTOMOBILE": r"automotive|\bauto\b|\bmechanics?\b|vehicle|dealership|collision|diesel",
    "CHEF": r"\bchef\b|\bcook\b|culinary|kitchen",
    "FINANCE": r"financ|investment|treasury|controller|\bfp&a\b",
    "APPAREL": r"apparel|fashion|clothing|garment|textile",
    "ENGINEERING": r"engineer",
    "ACCOUNTANT": r"accountant|accounting|bookkeep|\bcpa\b|auditor|accounts (?:payable|receivable)|"
                  r"\btax\b",
    "CONSTRUCTION": r"construction|carpenter|electrician|plumber|superintendent|foreman|laborer|"
                    r"\bhvac\b|estimator|roofer|mason\b",
    "PUBLIC-RELATIONS": r"public relations|\bpr\b|communications (?:manager|specialist|"
                        r"coordinator|director)|media relations|publicist",
    "BANKING": r"\bbank|teller|loan officer|mortgage|credit analyst|underwriter",
    "ARTS": r"\bartist|art director|museum|curator|musician|illustrator|\barts?\b",
    # "airport" was dropped: it matched locations (food jobs at an airport)
    "AVIATION": r"aviation|\bpilots?\b|aircraft|\bflight|airline|aerospace|avionics|\ba&p\b",
}
TITLE_RE = {k: re.compile(v, re.I) for k, v in TITLE_MAP.items()}

#: Exact LinkedIn industry names (mappings/industries.csv) per category.
INDUSTRY_MAP = {
    "HR": ["Human Resources Services", "Staffing and Recruiting"],
    "DESIGNER": ["Design Services", "Graphic Design", "Interior Design"],
    "INFORMATION-TECHNOLOGY": ["IT Services and IT Consulting", "Software Development",
                               "Computer and Network Security",
                               "Information Technology & Services",
                               "Computer Networking Products",
                               "Data Infrastructure and Analytics"],
    "TEACHER": ["Education", "Primary and Secondary Education", "Higher Education",
                "Education Administration Programs", "E-Learning Providers",
                "Education Management"],
    "ADVOCATE": ["Non-profit Organizations", "Civic and Social Organizations",
                 "Individual and Family Services"],
    "BUSINESS-DEVELOPMENT": [],
    "HEALTHCARE": ["Hospitals and Health Care", "Medical Practices", "Mental Health Care",
                   "Hospitals", "Individual and Family Services",
                   "Nursing Homes and Residential Care Facilities", "Home Health Care Services"],
    "FITNESS": ["Wellness and Fitness Services", "Health, Wellness & Fitness"],
    "AGRICULTURE": ["Farming", "Agriculture, Construction, Mining Machinery Manufacturing",
                    "Ranching", "Food and Beverage Manufacturing"],
    "BPO": ["Outsourcing and Offshoring Consulting", "Telecommunications"],
    "SALES": ["Retail", "Wholesale"],
    "CONSULTANT": ["Business Consulting and Services", "Management Consulting"],
    "DIGITAL-MEDIA": ["Online Audio and Video Media", "Broadcast Media Production and Distribution",
                      "Media Production", "Advertising Services", "Marketing Services",
                      "Entertainment Providers", "Online Media"],
    "AUTOMOBILE": ["Motor Vehicle Manufacturing", "Retail Motor Vehicles", "Automotive",
                   "Motor Vehicle Parts Manufacturing"],
    "CHEF": ["Restaurants", "Food and Beverage Services", "Hospitality"],
    "FINANCE": ["Financial Services", "Investment Management", "Investment Banking",
                "Venture Capital and Private Equity Principals", "Capital Markets"],
    "APPAREL": ["Retail Apparel and Fashion", "Apparel & Fashion", "Textile Manufacturing"],
    "ENGINEERING": ["Civil Engineering", "Engineering Services",
                    "Mechanical Or Industrial Engineering", "Industrial Machinery Manufacturing"],
    "ACCOUNTANT": ["Accounting"],
    "CONSTRUCTION": ["Construction", "Building Construction", "Civil Engineering",
                     "Architecture and Planning"],
    "PUBLIC-RELATIONS": ["Public Relations and Communications Services",
                         "Public Relations & Communications"],
    "BANKING": ["Banking", "Credit Intermediation"],
    "ARTS": ["Fine Art", "Performing Arts", "Museums, Historical Sites, and Zoos",
             "Artists and Writers", "Musicians"],
    "AVIATION": ["Airlines and Aviation", "Aviation and Aerospace Component Manufacturing",
                 "Airlines/Aviation", "Defense and Space Manufacturing"],
}
#: LinkedIn job-function tags (mappings/skills.csv `skill_name`) per category.
FUNCTION_MAP = {
    "HR": ["Human Resources"], "DESIGNER": ["Design"],
    "INFORMATION-TECHNOLOGY": ["Information Technology"], "TEACHER": ["Education", "Training"],
    "ADVOCATE": [], "BUSINESS-DEVELOPMENT": ["Business Development"],
    "HEALTHCARE": ["Health Care Provider"], "FITNESS": [], "AGRICULTURE": [],
    "BPO": ["Customer Service"], "SALES": ["Sales"], "CONSULTANT": ["Consulting"],
    "DIGITAL-MEDIA": ["Advertising", "Writing/Editing"], "AUTOMOBILE": [], "CHEF": [],
    "FINANCE": ["Finance"], "APPAREL": [], "ENGINEERING": ["Engineering"],
    "ACCOUNTANT": ["Accounting/Auditing"], "CONSTRUCTION": [],
    "PUBLIC-RELATIONS": ["Public Relations"], "BANKING": [], "ARTS": ["Art/Creative"],
    "AVIATION": [],
}


def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def en_stopword_share(t: str) -> float:
    toks = re.findall(r"[a-záéíóúñü]+", t.lower())
    return sum(1 for w in toks if w in STOP) / max(1, len(toks))


def invert(mapping: dict[str, list[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for cat, names in mapping.items():
        for n in names:
            out[n].add(cat)
    return out


def stage(raw: Path, out: Path) -> dict[str, int]:
    def read(rel: str) -> pd.DataFrame:
        return pd.read_csv(raw / "extracted" / rel, dtype=str, keep_default_na=False)

    postings = read("postings.csv")
    desc = postings["description"].str.strip()
    has_desc = desc != ""
    d = desc[has_desc]
    dup_group = d.map(lambda s: hashlib.sha1(norm_ws(s).casefold().encode()).hexdigest()[:16])
    lang = pd.Series(np.where(d.map(en_stopword_share) >= 0.05, "en", "und"), index=d.index)
    marker_hit = d.map(lambda t: bool(MARKER_ANY.search(t)))

    # Left joins on purpose: an industry id with no name in the mapping file stays in the list
    # (as '' -- 34 ids have empty names) rather than vanishing.
    ji = read("jobs/job_industries.csv").merge(read("mappings/industries.csv"),
                                               on="industry_id", how="left")
    js = read("jobs/job_skills.csv").merge(read("mappings/skills.csv"), on="skill_abr", how="left")
    benefits = read("jobs/benefits.csv")
    company_industries = read("companies/company_industries.csv")
    inds_by_job = ji.groupby("job_id")["industry_name"].agg(lambda x: sorted(set(map(str, x))))
    funcs_by_job = js.groupby("job_id")["skill_name"].agg(lambda x: sorted(set(map(str, x))))
    bens_by_job = benefits.groupby("job_id")["type"].agg(lambda x: sorted(set(x)))
    comp_ind = company_industries.groupby("company_id")["industry"].agg(lambda x: sorted(set(x)))
    ind2cat, f2cat = invert(INDUSTRY_MAP), invert(FUNCTION_MAP)

    other_cols = [c for c in postings.columns if c not in ("job_id", "title", "description")]
    rows = []
    kept = postings.loc[has_desc]
    for idx, rd in zip(kept.index, kept.to_dict("records")):
        job_id, text = rd["job_id"], d.at[idx]
        inds = inds_by_job.get(job_id, [])
        funcs = funcs_by_job.get(job_id, [])
        company_id = rd["company_id"]
        extra = {c: rd[c] for c in other_cols}
        extra.update({
            "industries": inds,
            "job_functions": funcs,
            "benefits": bens_by_job.get(job_id, []),
            "company_industries": (comp_ind.get(re.sub(r"\.0$", "", company_id), [])
                                   if company_id else []),
            "char_len": len(text),
            "in_400_6000": 400 <= len(text) <= 6000,
            "protected_marker_hit_jd": bool(marker_hit.at[idx]),
            "desc_dup_group": dup_group.at[idx],
            "kaggle_resume_cats_by_title": [k for k, rx in TITLE_RE.items()
                                            if rx.search(rd["title"])],
            "kaggle_resume_cats_by_industry": sorted({c for x in inds for c in ind2cat.get(x, ())}),
            "kaggle_resume_cats_by_function": sorted({c for x in funcs for c in f2cat.get(x, ())}),
        })
        rows.append({"job_id": str(job_id), "text": text, "title": rd["title"].strip(),
                     "occupation": ";".join(funcs), "lang": lang.at[idx], "split": "",
                     "source_file": "postings.csv", "extra": dumps(extra)})
    jobs = pd.DataFrame(rows)
    write_table(jobs, out / "jobs.parquet", compression="zstd")
    return {"jobs": len(jobs), "dropped_empty_description": int((~has_desc).sum())}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
