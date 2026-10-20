"""Stage the public part of PopResume: raw/ -> jobs.parquet (5 JDs).

    python -I scripts/external/stage_popresume.py <raw dir> <staged dir>

PopResume's resumes sit behind a manual-approval Hugging Face gate (see the blocker in
data/external_manifests/popresume.json), so only the five job descriptions in the public code
repository can be staged. They are worth staging anyway: they are part of the external holdout
(`external.holdout_index`), and a JD that is not on disk cannot be guarded against. All five are
under the project's 400-character JD minimum, so the pool filter would drop them in any case.

The resume text, once the gate is passed, is not staged by this script; the acquirer's draft
(`_scripts/popresume/stage_popresume_text.py`) was tested only on fabricated rows.
Reads raw/github_repo/2_scoring/job_info_v2/*_job.txt and
raw/github_repo/1_data_generation/sources/job_spec_deterministic.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _staging import main, write_table  # noqa: E402

SLUG = "popresume"
TABLES = ("jobs",)
REQUIRES = ()
JD_DIR = "github_repo/2_scoring/job_info_v2"
SPEC = "github_repo/1_data_generation/sources/job_spec_deterministic.json"
TITLES = {"accountants_auditors": "Accountants and Auditors",
          "construction_laborers": "Construction Laborers",
          "elementary_middle_school_teachers": "Elementary and Middle School Teachers",
          "registered_nurses": "Registered Nurses", "software_developers": "Software Developers"}
NOTE = "short JD used by the PopResume scoring script; below the project's 400-char JD minimum"


def stage(raw: Path, out: Path) -> dict[str, int]:
    spec = json.loads((raw / SPEC).read_text(encoding="utf-8"))
    rows = []
    for f in sorted((raw / JD_DIR).iterdir(), key=lambda p: p.name):
        if not f.name.endswith("_job.txt"):
            continue
        occ = f.name[: -len("_job.txt")]
        text = f.read_text(encoding="utf-8")
        s = spec.get(occ, {})
        rows.append({"job_id": f"popresume-jd-v2-{occ}", "text": text.strip(),
                     "title": TITLES.get(occ, occ), "occupation": occ, "lang": "en", "split": "",
                     "source_file": f"raw/{JD_DIR}/{f.name}",
                     "extra": json.dumps({"onet_soc_code": s.get("occupation_code"),
                                          "job_zone": s.get("job_zone"),
                                          "onet_skills": s.get("skills"),
                                          "job_info_version": "v2", "raw_len_chars": len(text),
                                          "stripped_len_chars": len(text.strip()), "note": NOTE})})
    jobs = pd.DataFrame(rows).astype("string")
    write_table(jobs, out / "jobs.parquet")
    return {"jobs": len(jobs)}


if __name__ == "__main__":
    main(stage, __doc__.splitlines()[0])
