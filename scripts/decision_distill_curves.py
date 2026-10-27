#!/usr/bin/env python
"""Decision-distillation training curves (docs/DECISION_DISTILL.md): the logged KL on attributed
rows and on attribute-free anchor rows, by optimizer step, read from each run's last
`trainer_state.json`, written as a CSV and a two-panel SVG. No plotting library is needed.

One epoch, so each logged value is on rows the model has not yet trained on.

    python scripts/decision_distill_curves.py --out $HBM_OUTPUT_ROOT/outputs/distill/curves \
        --run "s42=$HBM_OUTPUT_ROOT/outputs/distill/decision_distill_seed42" ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
PANELS = (("distill/attr_kl", "KL on attributed rows (to SFT-v3 on the attribute-free CV)"),
          ("distill/free_kl", "KL on attribute-free anchor rows (to SFT-v3 on the same CV)"))


def history(run_dir: Path) -> pd.DataFrame:
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: int(p.parent.name.split("-")[1]))
    if not states:
        raise FileNotFoundError(f"no checkpoint trainer_state.json under {run_dir}")
    rows = [h for h in json.loads(states[-1].read_text())["log_history"] if PANELS[0][0] in h]
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
    keep = ["label", "step", "learning_rate", "grad_norm"] + [
        c for c in table.columns if c.startswith("distill/")]
    table[keep].to_csv(args.out / "curves.csv", index=False)

    w, ph, left, top = 720, 200, 60, 30
    steps = table["step"].max()
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{2 * ph + 40}" '
           'font-family="sans-serif" font-size="11">',
           f'<rect width="{w}" height="{2 * ph + 40}" fill="white"/>']
    for p, (column, title) in enumerate(PANELS):
        y0 = top + p * ph
        ymax = max(float(table.groupby("label")[column].transform(
            lambda s: s.rolling(args.smooth, min_periods=1).mean()).max()), 1e-6) * 1.1

        def X(v):
            return left + v / steps * (w - left - 120)

        def Y(v, y0=y0, ymax=ymax):
            return y0 + ph - 40 - v / ymax * (ph - 60)

        out.append(f'<text x="{left}" y="{y0}" font-weight="bold">{title}, rolling mean of '
                   f'{args.smooth} logs</text>')
        out.append(f'<line x1="{left}" x2="{X(steps):.1f}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" '
                   'stroke="#999"/>')
        for t in (0, ymax / 2, ymax / 1.1):
            out.append(f'<text x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{t:.3f}</text>')
        for s in range(0, int(steps) + 1, 50):
            out.append(f'<text x="{X(s):.1f}" y="{Y(0) + 14:.1f}" text-anchor="middle">{s}</text>')
        for i, (label, g) in enumerate(table.groupby("label", sort=False)):
            v = g[column].rolling(args.smooth, min_periods=1).mean()
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(g["step"], v))
            color = COLORS[i % len(COLORS)]
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                       'stroke-width="1.5"/>')
            out.append(f'<text x="{X(steps) + 8:.1f}" y="{y0 + 20 + 14 * i}" fill="{color}">'
                       f'{label}</text>')
    out.append(f'<text x="{w / 2}" y="{2 * ph + 34}" text-anchor="middle">optimizer step</text>')
    out.append("</svg>")
    (args.out / "curves.svg").write_text("\n".join(out) + "\n")
    first = table.groupby("label").head(args.smooth).groupby("label")[[c for c, _ in PANELS]].mean()
    last = table.groupby("label").tail(args.smooth).groupby("label")[[c for c, _ in PANELS]].mean()
    print(pd.concat({"first": first, "last": last}, axis=1).round(4).to_string())


if __name__ == "__main__":
    main()
