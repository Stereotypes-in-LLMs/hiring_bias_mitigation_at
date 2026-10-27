#!/usr/bin/env python
"""GRPO rollout signal by step (docs/GRPO_IMPROVE.md, section 13): TRL's `frac_reward_zero_std`,
the share of groups whose rollouts all got one reward and so give no gradient, read from each
run's last `trainer_state.json`. Writes the per-step CSV, the window means the round-6
registration names (steps 1-50, 51-100, 101 to the end), and a one-panel SVG. No plotting
library is needed.

    python scripts/grpo_zero_variance.py \
        --out $HBM_OUTPUT_ROOT/outputs/grpo_improve/zero_variance_r6 \
        --run "R4 s42=$HBM_OUTPUT_ROOT/outputs/grpo/grpo_improve_r4_tipped_v2" \
        --run "R6 s42=$HBM_OUTPUT_ROOT/outputs/grpo/grpo_improve_r6_distill_seed42" ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLUMN = "frac_reward_zero_std"
WINDOWS = ((1, 50), (51, 100), (101, 10**6))
COLORS = ("#7a7a7a", "#1f77b4", "#bdbdbd", "#d62728", "#2ca02c", "#9467bd")


def history(run_dir: Path) -> pd.DataFrame:
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if (run_dir / "trainer_state.json").exists():
        states.append(run_dir / "trainer_state.json")
    if not states:
        raise FileNotFoundError(f"no trainer_state.json under {run_dir}")
    rows = [h for h in json.loads(states[-1].read_text())["log_history"] if COLUMN in h]
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smooth", type=int, default=5, help="rolling mean over this many logs")
    args = ap.parse_args()
    frames = []
    for spec in args.run:
        label, path = spec.split("=", 1)
        frames.append(history(Path(path)).assign(label=label))
    table = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    keep = ["label", "step", COLUMN] + [c for c in ("reward", "reward_std", "kl") if c in table]
    table[keep].to_csv(args.out / "zero_variance.csv", index=False)

    windows = []
    for label, g in table.groupby("label", sort=False):
        row = {"label": label, "steps": int(g["step"].max())}
        for lo, hi in WINDOWS:
            w = g[(g["step"] >= lo) & (g["step"] <= hi)]
            name = f"steps_{lo}_{hi if hi < 10**6 else 'end'}"
            row[name] = round(float(w[COLUMN].mean()), 4) if len(w) else None
        windows.append(row)
    windows = pd.DataFrame(windows)
    windows.to_csv(args.out / "windows.csv", index=False)
    print(windows.to_string(index=False))

    w, h, left, top = 720, 300, 60, 30
    steps = table["step"].max()

    def X(v):
        return left + v / steps * (w - left - 120)

    def Y(v):
        return top + (h - 70) * (1 - v)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
           'font-family="sans-serif" font-size="11">',
           f'<rect width="{w}" height="{h}" fill="white"/>',
           f'<text x="{left}" y="{top - 12}" font-weight="bold">Share of groups with zero reward '
           f'variance ({COLUMN}), rolling mean of {args.smooth} logs</text>']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{left}" x2="{X(steps):.1f}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" '
                   'stroke="#eee"/>')
        out.append(f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.2f}</text>')
    for s in range(0, int(steps) + 1, 50):
        out.append(f'<text x="{X(s):.1f}" y="{Y(0) + 14:.1f}" text-anchor="middle">{s}</text>')
    for i, (label, g) in enumerate(table.groupby("label", sort=False)):
        v = g[COLUMN].rolling(args.smooth, min_periods=1).mean()
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g["step"], v))
        color = COLORS[i % len(COLORS)]
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        out.append(f'<text x="{X(steps) + 8:.1f}" y="{top + 14 * i}" fill="{color}">{label}</text>')
    out.append(f'<text x="{w / 2}" y="{h - 10}" text-anchor="middle">optimizer step</text>')
    out.append("</svg>")
    (args.out / "zero_variance.svg").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
