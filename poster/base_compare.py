"""The poster's results table: the base model, SFT-v3 and the methods after it, against the base.

Same standard as every comparison here (``scripts/operating_point_control.py``): each arm at its
own greedy hire rate, the base model moved to the same hire count, 95% CI over candidates. The
rows are fixed below: DPO and GRPO (each method's recipe with three training seeds and the lowest
seed-mean change against SFT-v3), decision distillation, and GRPO trained on top of the
distilled model (one seed; DPO on top of distillation was never trained). Writes
``poster/base_compare.json``.

    HBM_OUTPUT_ROOT=$XW ../venv/bin/python poster/base_compare.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
import operating_point_control as opc  # noqa: E402

BASE_RUN = "Qwen3.5-9B--en--sft--adapter--en_only_cscs"   # its "base" scores: Qwen3.5-9B itself

# (arms.csv label or recipe name, row on the poster)
ROWS = [
    ("DPO_V3_4880", "+ DPO"),
    ("GRPO+decay R4", "+ GRPO"),
    ("Distill", "+ decision distillation"),
    ("GRPO from distill s42", "+ distillation, then GRPO"),
]


def rows(rel: str) -> list[dict]:
    with open(ROOT / rel, newline="") as f:
        return list(csv.DictReader(f))


def mean(items: list[dict], key: str) -> float:
    return sum(float(s[key]) for s in items) / len(items)


def main() -> None:
    base, _ = opc.load_scores(BASE_RUN, "base")
    arm_rows = rows("reports/final/arms.csv")
    arms = {r["label"]: r["run"] for r in arm_rows}
    parent = {r["label"]: r["parent"] for r in arm_rows}
    by_run = {r["run"]: r["label"] for r in arm_rows}
    table = {r["label"]: r for r in rows("reports/final/final_table.csv")}

    def gpu_h(label: str) -> float:
        """Extra GPU-h over SFT-v3; an arm trained on top of another arm adds that arm's
        training (their data is the same round-3 data, counted once)."""
        up = by_run.get(parent[label])
        extra = float(table[label]["extra_gpu_h"])
        return extra + (float(table[up]["train_gpu_h"]) if up and up != "SFT-v3" else 0.0)

    def against_base(run: str) -> dict:
        frame, _ = opc.load_scores(run, "adapter")
        c = opc.matched_comparison(base, frame)
        return {"run": run, "hire_pct": 100 * c["hire_rate_run"],
                "unstable_pct": 100 * c["unstable_run"],
                "base_unstable_pct": 100 * c["unstable_ref_matched"],
                "delta_pp": c["delta_matched_pp"], "ci": [c["ci_low"], c["ci_high"]]}

    hire0, unstable0 = opc.unstable_share(base, "margin", 0.0)
    out = {"source": "poster/base_compare.py, operating_point_control.matched_comparison",
           "base": {"run": BASE_RUN + " --base", "hire_pct": 100 * hire0,
                    "unstable_pct": 100 * unstable0},
           "sft_v3": against_base(arms["SFT-v3"]), "rows": []}
    for name, label in ROWS:
        labels = [lab for lab in arms
                  if lab == name or re.fullmatch(rf"{re.escape(name)} s\d+", lab)]
        if not labels:
            raise SystemExit(f"no arm for {name}")
        seeds = [against_base(arms[lab]) for lab in labels]
        out["rows"].append({
            "name": name, "label": label, "arms": labels, "seeds": seeds,
            "delta_pp_mean": mean(seeds, "delta_pp"),
            "unstable_pct_mean": mean(seeds, "unstable_pct"),
            "hire_pct_mean": mean(seeds, "hire_pct"),
            "delta_vs_sft_v3_pp_mean": mean([table[lab] for lab in labels], "delta_own_pp"),
            "extra_gpu_h": sum(gpu_h(lab) for lab in labels) / len(labels)})
    path = HERE / "base_compare.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path.name}: SFT-v3 {out['sft_v3']['delta_pp']:+.2f} pp; "
          + "; ".join(f"{r['name']} {r['delta_pp_mean']:+.2f}" for r in out["rows"]))


if __name__ == "__main__":
    main()
