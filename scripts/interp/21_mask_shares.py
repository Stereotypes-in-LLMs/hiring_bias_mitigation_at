"""Layer-masked check (docs/DISTILL_PATCHING.md): the share of the full adapter's shrink each
masked half reproduces, s = log r_half / log r_full, and the residual validity check.

    python scripts/interp/21_mask_shares.py --full distill43 \\
        --half distill43_l0_20 --half distill43_l21_31

Reads reports/interp/distill/<spec>/{sensitivity,residual}.csv, written by
20_distill_compare.py for each spec against the same reference; writes
reports/interp/distill/<full>/mask_shares.csv and mask_validity.csv. No GPU.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.interp.io import report_dir  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--full", required=True)
    ap.add_argument("--half", action="append", required=True)
    ap.add_argument("--max-depth", type=int, default=21,
                    help="depths up to which the residual is fixed by layers 0-20 alone")
    args = ap.parse_args()
    read = {s: pd.read_csv(report_dir("distill", s) / "sensitivity.csv").set_index("contrast")
            for s in [args.full, *args.half]}
    rows = []
    for c in read[args.full].index:
        for stat in ("r_scale", "r_attr"):
            row = {"contrast": c, "stat": stat, args.full: read[args.full].at[c, stat]}
            for h in args.half:
                row[h] = read[h].at[c, stat]
                row[f"s_{h}"] = np.log(read[h].at[c, stat]) / np.log(read[args.full].at[c, stat])
            row["s_sum"] = sum(row[f"s_{h}"] for h in args.half)
            rows.append(row)
    shares = pd.DataFrame(rows)
    out = report_dir("distill", args.full)
    shares.to_csv(out / "mask_shares.csv", index=False)
    print(shares.round(3).to_string(index=False))

    res = {s: pd.read_csv(report_dir("distill", s) / "residual.csv")
           for s in [args.full, *args.half]}
    checks = []
    for h in args.half:
        j = res[h].merge(res[args.full], on=["contrast", "depth"], suffixes=("", "_full"))
        j = j[j["depth"].between(1, args.max_depth)]
        checks.append({"half": h, "depths": f"1-{args.max_depth}",
                       "ratio_to_ref_min": j["ratio"].min(), "ratio_to_ref_max": j["ratio"].max(),
                       "ratio_to_full_min": (j["ratio"] / j["ratio_full"]).min(),
                       "ratio_to_full_max": (j["ratio"] / j["ratio_full"]).max()})
    checks = pd.DataFrame(checks)
    checks.to_csv(out / "mask_validity.csv", index=False)
    print(f"\nvalidity, residual depths 1-{args.max_depth}\n"
          + checks.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
