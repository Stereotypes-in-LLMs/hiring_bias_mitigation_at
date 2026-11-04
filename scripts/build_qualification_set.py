#!/usr/bin/env python
"""The identity x qualification test set: RJDB's dev split, never read before.

Each RJDB record is a job description with a matched CV and a GPT-4 weakened rewrite of it
(skills removed, experience shortened or dropped), so every triple carries a qualification
contrast under one job. The dev split (2,454 triples) was held out from every training set
(`holdout_splits` in data/external.py) and no audit has read it. This script takes the triples
whose weakening is effective and clean, samples a fixed number, and writes one pair per CV
version. `build_eval_set` then injects the identity cues as on the benchmark.

pair_id is `<candidate>__<job>__<matched|weakened>`: the matched comparison clusters its
bootstrap on the text before the first `__`, so both versions of a CV resample together.

    HBM_OUTPUT_ROOT=$XW python scripts/build_qualification_set.py [--n 300] [--seed 42]
    # -> $XW/artifacts/qualification_control/pairs_rjdb_dev.parquet, stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.data.external import parse_extra, prefixed_id  # noqa: E402
from hiring_bias_mitigation.data.external_eval import RJDB, _mentions  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

OUT = "artifacts/qualification_control"


def triples(cvs: pd.DataFrame, jobs: pd.DataFrame, pairs: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = cvs[cvs["split"] == split]
    matched = subset[subset["variant"] == ""].set_index("cv_id", drop=False)
    job_of = dict(zip(pairs["cv_id"], pairs["job_id"]))
    job_by_id = jobs.set_index("job_id", drop=False)
    rows = []
    for weak in subset[subset["variant"] != ""].to_dict("records"):
        base = matched.loc[weak["base_cv_id"]]
        job = job_by_id.loc[job_of[base["cv_id"]]]
        wextra, mextra = parse_extra(weak["extra"]), parse_extra(base["extra"])
        removed = [s.strip() for s in wextra.get("removed_skills") or [] if str(s).strip()]
        still = [s for s in removed if _mentions(s, weak["text"])]
        kind = wextra.get("experience_change_kind", "")
        rows.append({
            "triple": base["cv_id"].removesuffix("-m"),
            "candidate_id": prefixed_id(RJDB, base["cv_id"]),
            "job_id": prefixed_id(RJDB, job["job_id"]),
            "job_description": job["text"], "job_position": job.get("title", "") or "",
            "cv_matched": base["text"], "cv_weakened": weak["text"],
            "experience_change_kind": kind, "n_removed_skills": len(removed),
            "removed_skills": ";".join(removed), "removed_skill_still_mentioned": bool(still),
            "full_name_changed": bool(mextra.get("name")) and bool(wextra.get("name"))
                                 and mextra.get("name") != wextra.get("name"),
            "occupation": base.get("occupation", ""),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300, help="triples to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="dev")
    args = parser.parse_args()
    staged = resolve_output_path("external_data/rjdb/staged")
    cvs, jobs, pairs = (pd.read_parquet(staged / f"{n}.parquet") for n in ("cvs", "jobs", "pairs"))
    if (cvs.loc[cvs["split"] == args.split, "split"] != args.split).any():
        raise SystemExit("split mismatch")
    t = triples(cvs, jobs, pairs, args.split)
    # Effective and clean: something was really taken away, nothing removed is still named,
    # and the person's name did not drift (the verifier's rules in external_eval).
    effective = (t["experience_change_kind"].isin(["shortened", "dropped"]) | (t["n_removed_skills"] > 0))
    clean = ~t["removed_skill_still_mentioned"] & ~t["full_name_changed"]
    eligible = t[effective & clean]
    if len(eligible) < args.n:
        raise SystemExit(f"only {len(eligible)} eligible triples")
    rng = np.random.default_rng(args.seed)
    chosen = eligible.iloc[np.sort(rng.choice(len(eligible), size=args.n, replace=False))]
    rows = []
    for r in chosen.to_dict("records"):
        for quality, text in (("matched", r["cv_matched"]), ("weakened", r["cv_weakened"])):
            rows.append({
                "pair_id": f"{r['candidate_id']}__{r['job_id']}__{quality}",
                "candidate_id": r["candidate_id"], "job_id": r["job_id"],
                "cv": text, "job_description": r["job_description"],
                "job_position": r["job_position"], "qualification": quality,
                "triple": r["triple"], "experience_change_kind": r["experience_change_kind"],
                "n_removed_skills": r["n_removed_skills"], "removed_skills": r["removed_skills"],
                "occupation": r["occupation"], "source": f"rjdb:{args.split}",
                "reference_decision": "", "reference_feedback": "",
            })
    out = pd.DataFrame(rows)
    assert_no_leakage(out, "qualification_control")
    if out["pair_id"].duplicated().any() or out["job_id"].nunique() != args.n:
        raise SystemExit("pair ids or jobs are not unique")
    out_dir = resolve_output_path(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pairs_rjdb_{args.split}.parquet"
    out.to_parquet(path, index=False)
    stats = {
        "split": args.split, "triples_in_split": int(len(t)),
        "effective": int(effective.sum()), "clean": int(clean.sum()), "eligible": int(len(eligible)),
        "sampled_triples": args.n, "seed": args.seed, "pairs": int(len(out)),
        "experience_change_kind": chosen["experience_change_kind"].value_counts().to_dict(),
        "n_removed_skills": chosen["n_removed_skills"].value_counts().sort_index().to_dict(),
        "cv_words_matched_median": float(chosen["cv_matched"].str.split().str.len().median()),
        "cv_words_weakened_median": float(chosen["cv_weakened"].str.split().str.len().median()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
    }
    (out_dir / f"stats_rjdb_{args.split}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
