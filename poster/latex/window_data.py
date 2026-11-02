"""Summarize the completed, registered window-LoRA parameter-budget control.

Reads per-seed comparisons; never averages confidence limits or runs scoring.
The old and matched window gaps use the same full-arm operating point.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def summarize(root=ROOT):
    sources, records, budgets = {}, [], []

    def read(relative):
        path = root / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    for seed in (42, 43, 44):
        prefix = "reports/experiment_followup/"
        budget = read(f"{prefix}window_budget_seed{seed}_after.json")
        done = read(f"{prefix}window_evaluation_seed{seed}.json")
        result = read(f"{prefix}window_seed{seed}/comparison.json")
        assert done["completed"] and done["seed"] == budget["seed"] == result["training_seed"] == seed
        assert set(done["comparisons"]) == {"sft_v3", "same_seed_full_layer"}
        assert budget["saved_parameters"] == budget["new_window_expected_parameters"] == 58085376
        assert budget["full_parameters"] == 58195968 and budget["old_window_parameters"] == 11403264
        assert abs(budget["relative_parameter_gap_pct"]) <= budget["budget_tolerance_pct"] == .2
        assert budget["rank"] == 163 and budget["layers"] == [15, 20]
        assert result["standard"] == "op-margin-v2"
        assert result["bootstrap"] == {"draws": 1000, "seed": 42, "unit": "candidate", "cuts": "fixed"}
        assert result["primary_sign"] == "window minus full; positive is worse"
        assert result["primary_operating_point"] == "same-seed full-layer r32 arm's own greedy hire count"
        for arm, suffix in (("full_r32", f"decision_distill_seed{seed}"),
                            ("window_r32", f"decision_distill_l15_20_seed{seed}"),
                            ("window_r163", f"decision_distill_l15_20_r163_seed{seed}")):
            assert result["inputs"][arm]["run"] == f"Qwen3.5-9B--en--sft--adapter--en_only_{suffix}"
        assert result["inputs"]["sft_v3"]["run"].endswith("en_only_sft_v3_additional")
        assert {x["prompts"]["n"] for x in result["inputs"].values()} == {31050}
        assert len({x["prompts"]["sha1"] for x in result["inputs"].values()}) == 1
        primary = {r["arm"]: r for r in result["primary"]}
        common = {r["arm"]: r for r in result["common"]}
        assert set(primary) == {"window_r32", "window_r163"}
        assert set(common) == {"full_r32", "window_r32", "window_r163"}
        for r in [*primary.values(), *common.values()]:
            assert r["prompts"] == 31050 and r["sets"] == 2700
            assert math.isfinite(r["ci_low"]) and r["ci_low"] <= r["ci_high"]
        assert primary["window_r32"]["hire_rate_pct"] == primary["window_r163"]["hire_rate_pct"]
        assert {r["hire_count"] for r in common.values()} == {6990}
        records.append({"seed": seed, "primary": primary, "common": common})
        budgets.append(budget)

    means = {
        "primary": {a: mean(r["primary"][a]["delta_window_minus_full_pp"] for r in records)
                    for a in ("window_r32", "window_r163")},
        "common": {a: mean(r["common"][a]["delta_arm_minus_full_pp"] for r in records)
                   for a in ("window_r32", "window_r163")},
    }
    ci_above = {basis: sum(r[basis]["window_r163"]["ci_low"] > 0 for r in records)
                for basis in ("primary", "common")}
    return {"complete": True, "seeds": [42, 43, 44], "parameters": {
                "full_r32": budgets[0]["full_parameters"],
                "window_r32": budgets[0]["old_window_parameters"],
                "window_r163": budgets[0]["saved_parameters"],
                "relative_gap_pct": budgets[0]["relative_parameter_gap_pct"]},
            "primary_operating_point": result["primary_operating_point"],
            "common_hire_count": 6990, "bootstrap": result["bootstrap"],
            "seed_means_pp": means, "matched_seed_cis_above_zero": ci_above,
            "aggregate_confidence_interval": None, "rows": records, "source_sha256": sources}


def write(root=ROOT):
    result = summarize(root)
    out = root / "reports/poster_followup/window_budget_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    primary = result["seed_means_pp"]["primary"]
    macros = {"WindowBudgetOldGap": f"{primary['window_r32']:+.2f}",
              "WindowBudgetMatchedGap": f"{primary['window_r163']:+.2f}",
              "WindowBudgetCommonGap": f"{result['seed_means_pp']['common']['window_r163']:+.2f}",
              "WindowBudgetOldParams": f"{result['parameters']['window_r32']/1e6:.1f}",
              "WindowBudgetMatchedParams": f"{result['parameters']['window_r163']/1e6:.1f}",
              "WindowBudgetFullParams": f"{result['parameters']['full_r32']/1e6:.1f}",
              "WindowBudgetCIAbove": str(result['matched_seed_cis_above_zero']['primary'])}
    text = "% Completed parameter-budget control; generated by window_data.py.\n"
    text += "\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in macros.items()) + "\n"
    (root / "poster/latex/window_data.tex").write_text(text)
    return result


if __name__ == "__main__":
    result = write()
    print(json.dumps({k: result[k] for k in ("parameters", "seed_means_pp", "matched_seed_cis_above_zero")}, indent=2))
