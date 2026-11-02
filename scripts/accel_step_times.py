"""Per-step times of the acceleration benchmarks B2 and B3 (docs/ACCEL_BENCH.md), from their logs.

The trainer's ``train_runtime`` (reports/cost/steps.csv) is one number per run; the logs also
hold each run's progress bar (B2, cumulative seconds at every optimizer step) and the GRPO
trainer's per-step ``step_time`` and ``completions/mean_length`` (B3). Written to
``reports/throughput_b/accel/step_times.csv``, one row per run and step, and read by
poster/latex/latex_data.py.

    HBM_OUTPUT_ROOT=$XW ../venv/bin/python scripts/accel_step_times.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOGS = Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw")) / "logs"
B2 = LOGS / "xw-bench-train-5236800.out"
B3 = LOGS / "xw-bench-grpo-5236801.out"
OUT = ROOT / "reports/throughput_b/accel/step_times.csv"
# A run's lines end at its cost record: "cost: train/<label> ok in ..."
COST = re.compile(r"cost: train/(\S+) ok in")


def segments(text: str) -> list[tuple[str, str]]:
    """(label, the log text of that run), in log order."""
    out, start = [], 0
    for m in COST.finditer(text):
        out.append((m.group(1), text[start:m.start()]))
        start = m.end()
    return out


def clock(t: str) -> int:
    parts = [int(v) for v in t.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts)))


def b2_rows() -> list[dict]:
    rows = []
    for label, seg in segments(B2.read_text(errors="replace")):
        seen = {}
        for step, total, elapsed in re.findall(r"(\d+)/(\d+) \[(\d+:\d{2}(?::\d{2})?)<", seg):
            if int(total) == 40 and int(step) > 0:
                seen[int(step)] = clock(elapsed)  # the bar repeats a step; the last print wins
        assert sorted(seen) == list(range(1, 41)), (label, sorted(seen)[:5], len(seen))
        rows += [{"bench": "B2", "label": label, "step": s, "elapsed_s": seen[s]} for s in seen]
    return rows


def b3_rows() -> list[dict]:
    rows = []
    for label, seg in segments(B3.read_text(errors="replace")):
        times = [float(v) for v in re.findall(r"'step_time': '([0-9.]+)'", seg)]
        lengths = [float(v) for v in re.findall(r"'completions/mean_length': '([0-9.]+)'", seg)]
        assert len(times) == 6 and len(lengths) >= 6, (label, len(times), len(lengths))
        rows += [{"bench": "B3", "label": label, "step": i + 1, "step_time_s": t,
                  "completion_tokens_mean": n} for i, (t, n) in enumerate(zip(times, lengths))]
    return rows


def main() -> None:
    table = pd.DataFrame(b2_rows() + b3_rows())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    b2 = table[table["bench"] == "B2"].pivot(index="step", columns="label", values="elapsed_s")
    for label in b2:
        t = b2[label]
        early, late = t[5] / 5, (t[40] - t[5]) / 35
        print(f"{label}: steps 1-5 {early:.1f} s/step, steps 6-40 {late:.1f} s/step")
    b3 = table[table["bench"] == "B3"].groupby("label")
    print(b3[["step_time_s", "completion_tokens_mean"]].mean().round(2))


if __name__ == "__main__":
    main()
