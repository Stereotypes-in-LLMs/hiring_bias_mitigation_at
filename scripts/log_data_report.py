"""Puts a data batch's direction check (D1-D7) and sample pairs on W&B as a data-check run.

    python scripts/log_data_report.py --dir artifacts/semisynthetic-v2/onpolicy-hire32/balanced

Reads direction_report.json (written by check_preference_direction.py), and report.json /
balance.json / dpo_v2_candidates.parquet when present. Summary keys are the report flattened
(generation/ and balance/ prefixes for the other two); tables: one row per verdict, and a
seeded sample of pairs with the judge's scores.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.generation import direction_check as DC  # noqa: E402
from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402

SAMPLE_COLUMNS = ("protected_group", "protected_attr", "condition", "reference_decision",
                  "chosen", "rejected", "judge_plausibility", "chosen_judge_plausibility")


def _json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def log_dir(name: str, project: str, n_samples: int, seed: int) -> None:
    import wandb

    root = Path(resolve_output_path(name))
    report = _json(root / "direction_report.json")
    if report is None:
        raise SystemExit(f"{root} has no direction_report.json; run check_preference_direction.py")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    tag = name.removeprefix("artifacts/").replace("/", "-")
    run = wandb.init(project=project, job_type="data-check", name=f"data-{tag}",
                     config={"dir": name, "commit": commit})
    run.summary.update(DC.flatten_report(report))
    for prefix, extra in (("generation", _json(root / "report.json")),
                          ("balance", _json(root / "balance.json"))):
        if extra:
            run.summary.update({f"{prefix}/{k}": v for k, v in DC.flatten_report(extra).items()})
    rows = [list(r) for r in DC.check_rows(report)]
    tables = {"checks": wandb.Table(columns=["scope", "check", "metric", "value", "pass"],
                                    data=rows)}
    candidates = root / "dpo_v2_candidates.parquet"
    if candidates.exists():
        pairs = pd.read_parquet(candidates)
        keep = [c for c in SAMPLE_COLUMNS if c in pairs.columns]
        sample = pairs.sample(n=min(n_samples, len(pairs)), random_state=seed)[keep]
        tables["samples"] = wandb.Table(dataframe=sample.astype(str))
    run.log(tables)
    run.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", action="append", required=True, help="a batch directory")
    parser.add_argument("--project", default="hiring-bias-mitigation")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for name in args.dir:
        log_dir(name, args.project, args.samples, args.seed)


if __name__ == "__main__":
    main()
