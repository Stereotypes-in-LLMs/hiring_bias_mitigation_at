"""Stage O*NET 31.0 (text release): raw/ -> occupations.parquet, job_titles.parquet.

    python -I scripts/external/stage_onet.py <raw dir> <staged dir>

O*NET is a rubric source, not CVs or JDs, so it does not use the cvs/jobs/pairs layout:

    occupations.parquet  one row per O*NET-SOC code: title, description, Job Zone (1-5, null
                         for 93 codes), task counts, title counts, and which skill/knowledge/
                         education tables cover it
    job_titles.parquet   every title string O*NET maps to a code (the O*NET title, the 'Job
                         Titles' list and the 'Sample of Reported Titles'), for title -> SOC
                         lookups. The 'Short Title' sentinel 'n/a' is written as ''.

Licence: CC BY 4.0 with a required USDOL/ETA attribution and change notice wherever derived
content ships (docs/EXTERNAL_DATA.md). Reads raw/extracted/db_31_0_text/*.txt
(tab-delimited, no quoting).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import main, write_table  # noqa: E402

SLUG = "onet"
TABLES = ("occupations", "job_titles")
REQUIRES = ()
TEXT_DIR = "extracted/db_31_0_text"
RELEASE = "O*NET 31.0 (August 2026)"
CODE = "O*NET-SOC Code"


def stage(raw: Path, out: Path) -> dict[str, int]:
    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(raw / TEXT_DIR / name, sep="\t", dtype=str, keep_default_na=False,
                           quoting=csv.QUOTE_NONE, encoding="utf-8")

    occ = read("Occupation Data.txt")
    jz = read("Job Zones.txt")
    tasks = read("Task Statements.txt")
    titles = read("Job Titles.txt")
    reported = read("Sample of Reported Titles.txt")
    has_ess = set(read("Essential Skills.txt")[CODE])
    has_kn = set(read("Knowledge.txt")[CODE])
    has_edu = set(read("Education.txt")[CODE])

    job_zone = dict(zip(jz[CODE], jz["Job Zone"].astype(int)))
    n_tasks = tasks.groupby(CODE).size().to_dict()
    n_core = tasks[tasks["Task Type"] == "Core"].groupby(CODE).size().to_dict()
    n_titles = titles.groupby(CODE).size().to_dict()
    n_reported = reported.groupby(CODE).size().to_dict()
    rows = []
    for r in occ.itertuples(index=False):
        c = r[0]
        rows.append({
            "onetsoc_code": c, "title": r[1].strip(), "description": r[2].strip(),
            "job_zone": job_zone.get(c), "n_tasks": int(n_tasks.get(c, 0)),
            "soc_2018_code": c[:7], "soc_major_group": c[:2],
            "n_core_tasks": int(n_core.get(c, 0)), "n_job_titles": int(n_titles.get(c, 0)),
            "n_reported_titles": int(n_reported.get(c, 0)),
            "extra": json.dumps({"source_file": "Occupation Data.txt", "release": RELEASE,
                                 "has_essential_skills": c in has_ess,
                                 "has_knowledge": c in has_kn,
                                 "has_education": c in has_edu}, ensure_ascii=False),
        })
    occupations = pd.DataFrame(rows)
    occupations["job_zone"] = occupations["job_zone"].astype("Int64")

    job_titles = pd.concat([
        pd.DataFrame({"onetsoc_code": occ[CODE], "title_kind": "onet_title",
                      "job_title": occ["Title"].str.strip(), "short_title": "", "sources": "",
                      "in_my_next_move": ""}),
        pd.DataFrame({"onetsoc_code": titles[CODE], "title_kind": "job_title",
                      "job_title": titles["Job Title"].str.strip(),
                      "short_title": titles["Short Title"].replace("n/a", ""),
                      "sources": titles["Source(s)"], "in_my_next_move": ""}),
        pd.DataFrame({"onetsoc_code": reported[CODE], "title_kind": "reported_title",
                      "job_title": reported["Reported Job Title"].str.strip(), "short_title": "",
                      "sources": "", "in_my_next_move": reported["Shown in My Next Move"]}),
    ], ignore_index=True)
    write_table(occupations, out / "occupations.parquet")
    write_table(job_titles, out / "job_titles.parquet")
    return {"occupations": len(occupations), "job_titles": len(job_titles)}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
