"""Rebuild the chart payload from the shipped aggregate CSVs. No model runs."""
from pathlib import Path
import csv
import json

root = Path(__file__).resolve().parent / "dist"
def read(name):
    with (root / "data" / name).open() as stream:
        return list(csv.DictReader(stream))
curves, final, overlap = read("curves.csv"), read("final_table.csv"), read("overlap.csv")
reference = {float(r["hire_rate_pct"]): float(r["unstable_pct"])
             for r in curves if r["label"] == "SFT-v3"}
rates = sorted(reference)
assert len(rates) == 51
data = {"rates": rates, "methods": {}, "overlap": []}
for method, prefix, label in [("dpo", "DPO_V3_4880", "DPO"),
                               ("grpo", "GRPO+decay R4", "GRPO + decay"),
                               ("distill", "Distill", "Self-distillation")]:
    seeds, intervals = [], []
    for seed in (42, 43, 44):
        rows = {float(r["hire_rate_pct"]): float(r["unstable_pct"])
                for r in curves if r["label"] == f"{prefix} s{seed}"}
        assert set(rows) == set(reference)
        seeds.append([rows[x] - reference[x] for x in rates])
        row = next(r for r in final if r["label"] == f"{prefix} s{seed}")
        intervals.append({"seed": seed, "delta": float(row["delta_common_pp"]),
                          "low": float(row["ci_common_low"]), "high": float(row["ci_common_high"])})
    data["methods"][method] = {"seeds": seeds, "mean": [sum(v)/3 for v in zip(*seeds)], "intervals": intervals}
    for row in overlap:
        if row["recipe"] == prefix and row["seeds"] in ("42/43", "42/44", "43/44"):
            data["overlap"].append({"method": label, "seeds": row["seeds"],
                                    "actual": float(row["jaccard"]), "control": float(row["sft_v3_jaccard"])})
assert len(data["overlap"]) == 9
(root / "results.js").write_text("window.AUDIT_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")
