#!/usr/bin/env python
"""One row per run: the standard's matched change with its 95% interval and the TPR change,
read from reports/reference_metrics.csv, drawn as a dot-and-interval SVG beside a CSV.

    python scripts/grpo_improve_seed_plot.py --out $HBM_OUTPUT_ROOT/outputs/grpo_improve/seeds \
        --run "R3 s42=Qwen3.5-9B--en--grpo--adapter--en_only_grpo_improve_r3_wsd" ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_NAME")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    table = pd.read_csv(REPO / "reports/reference_metrics.csv").set_index("run")
    rows = []
    for spec in args.run:
        label, run = spec.split("=", 1)
        r = table.loc[run]
        rows.append({"label": label, "run": run, "delta_pp": r["primary_delta_pp"],
                     "ci_low_pp": r["primary_ci_low_pp"], "ci_high_pp": r["primary_ci_high_pp"],
                     "tpr_vs_ref_pp": r["std_tpr_vs_ref_pp"], "verdict": r["std_verdict"]})
    frame = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "seeds.csv", index=False)
    w, row_h, left, right = 720, 26, 210, 40
    h = 60 + row_h * len(frame)
    lo, hi = min(frame["ci_low_pp"].min(), -2.0), max(frame["ci_high_pp"].max(), 2.2)

    def X(v):
        return left + (v - lo) / (hi - lo) * (w - left - right)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           f'font-family="sans-serif" font-size="11"><rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{left}" y="16" font-weight="bold">Matched change in unstable sets vs '
           'SFT-v3, pp [95% CI]</text>',
           f'<line x1="{X(0):.1f}" x2="{X(0):.1f}" y1="24" y2="{h - 24}" stroke="#999" '
           'stroke-dasharray="4 3"/>']
    for t in range(int(lo), int(hi) + 1):
        out.append(f'<text x="{X(t):.1f}" y="{h - 8}" text-anchor="middle">{t:+d}</text>')
    for i, r in frame.iterrows():
        y = 36 + row_h * i
        color = "#1f77b4" if r["delta_pp"] < 0 else "#d62728"
        out.append(f'<text x="{left - 8}" y="{y + 4}" text-anchor="end">{r["label"]}</text>')
        out.append(f'<line x1="{X(r["ci_low_pp"]):.1f}" x2="{X(r["ci_high_pp"]):.1f}" '
                   f'y1="{y}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        out.append(f'<circle cx="{X(r["delta_pp"]):.1f}" cy="{y}" r="4" fill="{color}"/>')
        out.append(f'<text x="{w - right + 4}" y="{y + 4}" fill="#555">TPR '
                   f'{r["tpr_vs_ref_pp"]:+.1f}</text>')
    out.append("</svg>")
    (args.out / "seeds.svg").write_text("\n".join(out) + "\n")
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
