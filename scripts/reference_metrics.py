"""Writes the per-run reference metrics: one row per audited run.

    python scripts/reference_metrics.py --run <run_name>   # one run, from outputs/raw/<run>.parquet
    python scripts/reference_metrics.py --config <audit cfg> --show   # the run of an audit config
    python scripts/reference_metrics.py --all              # every scored run with a raw parquet
    python scripts/reference_metrics.py --table-only       # rebuild reports/reference_metrics.csv
    python scripts/reference_metrics.py --show-only <run>  # print a run's verdict from the table

Rows go to reports/reference_metrics/<run>.json and are collected into
reports/reference_metrics.csv. scripts/run_audit.py writes a run's row after every audit; run
this again after an op-margin-v2 margin pass, which fills the confidence columns. What each
column means: docs/METRICS.md, "Reference metrics per run". How they sit beside the comparison
standard: docs/EVAL_STANDARD.md. The rows read eval/results/, the raw parquets and the margin
files only, so this is CPU work and safe while a sweep runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.analysis import reference_metrics as RM  # noqa: E402
from hiring_bias_mitigation.eval.runner import build_run_name  # noqa: E402
from hiring_bias_mitigation.utils.config import (
    REPO_ROOT,  # noqa: E402
    load_config,  # noqa: E402
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("reference_metrics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--run", nargs="+", help="run names, as in eval/results/<run>.json")
    which.add_argument("--config", nargs="+", help="audit configs; their run names")
    which.add_argument("--all", action="store_true", help="every scored run with a raw parquet")
    which.add_argument("--table-only", action="store_true", help="only rebuild the CSV")
    which.add_argument("--show-only", nargs="+", metavar="RUN",
                       help="print these runs' verdicts from the table, compute nothing")
    parser.add_argument("--show", action="store_true", help="print the verdicts after writing")
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "eval" / "results"))
    args = parser.parse_args()

    if args.show_only:
        show(args.show_only)
        return
    if args.table_only:
        log.info("wrote %s", RM.write_table())
        return
    results = Path(args.results_dir)
    names = args.run or ([build_run_name(load_config(c)) for c in args.config]
                         if args.config else None)
    paths = ([results / f"{name}.json" for name in names] if names
             else sorted(p for p in results.glob("*.json") if p.name != "index.json"))
    written, skipped = 0, []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if "summary" not in record:
            continue
        out = RM.write_for_record(record, results_dir=results)
        if out is None:
            skipped.append(record["run_name"])
        else:
            written += 1
            log.info("wrote %s", out)
    log.info("%d row(s) written, table %s", written, RM.TABLE)
    if skipped:
        log.info("%d run(s) skipped, no raw parquet under outputs/raw/: %s", len(skipped),
                 ", ".join(skipped[:8]) + (" ..." if len(skipped) > 8 else ""))
    if args.show and names:
        show(names)


def show(names: list[str]) -> None:
    """Prints each run's guardrails and verdict from reports/reference_metrics.csv."""
    import pandas as pd

    table = pd.read_csv(RM.TABLE).set_index("run")
    columns = ["decided_pct", "std_reference_arm", "primary_delta_pp", "primary_ci_low_pp",
               "primary_ci_high_pp", "merge_kept_pct", "std_tpr_vs_ref_pp",
               "std_agree_vs_ref_pp", "guard_merge", "guard_tpr", "guard_agree", "guard_decided",
               "std_verdict"]
    for name in names:
        if name not in table.index:
            print(f"{name}: no row in {RM.TABLE}")
            continue
        print(name)
        for column in columns:
            print(f"  {column:22s} {table.at[name, column]}")


if __name__ == "__main__":
    main()
