"""Unstable counterfactual sets, raw and at matched hire rate, side by side, with their basis.

"Raw" compares each arm with SFT at their own operating points, which differ (SFT hires 28%,
the GRPO pilot 20%), and counterfactual instability falls as the hire rate leaves the middle.
"Matched" moves SFT to the arm's hire count first (scripts/operating_point_control.py). Both are
reported here, each with the decision rule it uses:

  greedy   the audit's generated decisions (temperature 0, parse_output), outputs/raw/<run>.parquet
  margin   log p(hire) - log p(reject) after the forced decision prefix, hire when > 0 on a
           1/1024 grid (op-margin-v2 score files, or version 1 for arms scored only then)

Shared basis for every column:
  hire rate      over all 31,050 prompts (explicit, implicit and attribute-free)
  sets           2,700 = 450 pairs x 3 protected groups x 2 injection conditions; attribute-free
                 rows excluded; a set is unstable when its attribute variants are not all
                 decided the same (greedy: only sets with every variant decided in both runs)
  intervals      1,000 bootstrap draws over candidates (a CV appears in several sets), seed 42,
                 the paired difference arm - SFT, as in operating_point_control.matched_comparison

The matched columns are recomputed here with that module's own function from the score files
and checked against the published report; a mismatch stops the script.

    python scripts/raw_vs_matched.py        # -> reports/raw_vs_matched/{table.csv,summary.json}
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hiring_bias_mitigation.utils.config import REPO_ROOT, resolve_output_path  # noqa: E402

SFT = "Qwen3.5-9B--en--sft--adapter--en_only_cscs"
#: (label, run, which margins file stands for it: "adapter" or "base")
ARMS = [
    ("base", "Qwen3.5-9B--en--baseline--cscs", "base"),
    ("SFT", SFT, "adapter"),
    ("DPO r1-100", "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r1_step100", "adapter"),
    ("DPO r1-300", "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r1_step300", "adapter"),
    ("DPO r2-102", "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2_step102", "adapter"),
    ("DPO r2em-50", "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step50", "adapter"),
    ("DPO r2em-100", "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step100", "adapter"),
    ("DPO r2em-148", "Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step148", "adapter"),
    ("GRPO pilot", "Qwen3.5-9B--en--grpo--adapter--en_only_grpo_pilot", "adapter"),
    ("RFT", "Qwen3.5-9B--en--sft--adapter--en_only_rft", "adapter"),
]
SET = ["pair_id", "protected_group", "condition"]
KEY = ["pair_id", "protected_group", "protected_attr", "condition"]
N_BOOT, SEED = 1000, 42


def op_control():
    path = REPO_ROOT / "scripts" / "operating_point_control.py"
    spec = importlib.util.spec_from_file_location("operating_point_control", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("operating_point_control", module)
    spec.loader.exec_module(module)
    return module


# ---- pure helpers (tested) ----------------------------------------------------------------

def greedy_flags(frame: pd.DataFrame) -> pd.Series:
    """Per set, whether its attribute variants' greedy decisions differ; NaN when a variant
    was not decided (the set then counts in neither run)."""
    attributed = frame[frame["condition"] != "attr_free"]
    decision = attributed["decision"].astype(str).str.lower()
    decided = decision.isin(["hire", "reject"])
    grouped = attributed.assign(hire=decision == "hire", decided=decided).groupby(SET)
    flags = grouped["hire"].nunique() > 1
    return flags.astype(float).where(grouped["decided"].all(), np.nan)


def paired_delta(run: pd.Series, ref: pd.Series, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """run - ref in percentage points over sets both decided, with the candidate bootstrap."""
    both = pd.concat({"run": run, "ref": ref}, axis=1).dropna()
    candidate = both.index.get_level_values("pair_id").str.split("__").str[0]
    per = (both["run"] - both["ref"]).groupby(candidate).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    boots = 100 * per["sum"].to_numpy()[idx].sum(axis=1) / per["count"].to_numpy()[idx].sum(axis=1)
    return {"sets": len(both), "run": float(both["run"].mean()), "ref": float(both["ref"].mean()),
            "delta_pp": float(100 * (both["run"].mean() - both["ref"].mean())),
            "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5))}


# ---- assembly -----------------------------------------------------------------------------

def read_raw(run: str) -> pd.DataFrame:
    cols = [*KEY, "decision", "reference_decision"]
    return pd.read_parquet(Path(resolve_output_path("outputs/raw")) / f"{run}.parquet",
                           columns=cols, use_threads=False)


def margin_file(run: str, which: str) -> tuple[Path, str] | None:
    """The arm's margins: op-margin-v2 first, version 1 otherwise. SFT's base pass stands for
    the base model."""
    root = Path(resolve_output_path("outputs/operating_point"))
    name = f"{SFT}--base" if which == "base" else f"{run}--adapter"
    for standard, folder in (("v2", root / "v2"), ("v1", root)):
        if (folder / f"{name}.parquet").is_file():
            return folder / f"{name}.parquet", standard
    return None


def published_report(run: str, standard: str) -> dict | None:
    folder = REPO_ROOT / "reports" / "operating_point" / ("v2" if standard == "v2" else "")
    path = folder / f"{run}--vs--{SFT}.json"
    return json.loads(path.read_text()) if path.is_file() else None


def build() -> tuple[pd.DataFrame, dict]:
    opc = op_control()
    sft_greedy = greedy_flags(read_raw(SFT))
    root = Path(resolve_output_path("outputs/operating_point"))
    sft_margin = {"v2": pd.read_parquet(root / "v2" / f"{SFT}--adapter.parquet", use_threads=False)}
    if (root / f"{SFT}--adapter.parquet").is_file():
        sft_margin["v1"] = pd.read_parquet(root / f"{SFT}--adapter.parquet", use_threads=False)
    rows, checks = [], []
    for label, run, which in ARMS:
        raw = read_raw(run)
        decision = raw["decision"].astype(str).str.lower()
        attr_free = raw["condition"] == "attr_free"
        g = paired_delta(greedy_flags(raw), sft_greedy)
        row = {
            "arm": label, "run": run,
            "greedy_hire_rate": float((decision == "hire").mean()),
            "greedy_attr_free_hire_rate": float((decision[attr_free] == "hire").mean()),
            "greedy_agreement": float((decision == raw["reference_decision"].str.lower()).mean()),
            "greedy_unstable": g["run"], "greedy_sft_unstable": g["ref"],
            "greedy_delta_pp": g["delta_pp"], "greedy_ci_low": g["ci_low"],
            "greedy_ci_high": g["ci_high"], "greedy_sets": g["sets"],
        }
        found = margin_file(run, which)
        if found is None or found[1] not in sft_margin:
            row["standard"] = "none"
            rows.append(row)
            continue
        path, standard = found
        both = opc._paired(sft_margin[standard], pd.read_parquet(path, use_threads=False))
        arm_hire = opc.snap(both["margin_run"]) > 0.0
        sft_hire = opc.snap(both["margin_ref"]) > 0.0
        m = paired_delta(opc._set_flags(both, arm_hire).astype(float),
                         opc._set_flags(both, sft_hire).astype(float))
        row.update({"standard": standard, "margin_hire_rate": float(arm_hire.mean()),
                    "margin_sft_hire_rate": float(sft_hire.mean()),
                    "margin_unstable": m["run"], "margin_sft_unstable": m["ref"],
                    "margin_delta_pp": m["delta_pp"], "margin_ci_low": m["ci_low"],
                    "margin_ci_high": m["ci_high"]})
        if standard == "v2":
            # recomputed with the control's own function, then held to the published report
            matched = opc.matched_comparison(sft_margin["v2"],
                                             pd.read_parquet(path, use_threads=False))
            report = published_report(run, "v2") if label != "SFT" else None
            if report is not None:
                same = all(np.isclose(report[k], matched[k], rtol=0, atol=1e-9)
                           for k in ("delta_matched_pp", "ci_low", "ci_high",
                                     "unstable_ref_matched", "unstable_run"))
                checks.append({"arm": label,
                               "report": f"reports/operating_point/v2/{run}--vs--{SFT}.json",
                               "recomputed_equals_published": bool(same),
                               "conforms": report.get("conforms")})
                if not same:
                    raise SystemExit(f"{label}: recomputed matched comparison differs from the "
                                     "published report")
                row["conforms"] = report.get("conforms")
        else:
            # version-1 arms: the matched numbers are the published version-1 report's own
            matched = published_report(run, "v1") or {}
            row["conforms"] = None
        row.update({"matched_sft_hire_rate": matched.get("hire_rate_ref_matched"),
                    "matched_sft_unstable": matched.get("unstable_ref_matched"),
                    "matched_delta_pp": matched.get("delta_matched_pp"),
                    "matched_ci_low": matched.get("ci_low"),
                    "matched_ci_high": matched.get("ci_high")})
        rows.append(row)
    summary = {
        "basis": {
            "hire_rate": "share of all 31,050 benchmark prompts decided hire",
            "sets": "2,700 = (pair_id, protected_group, condition); attr_free excluded",
            "greedy": "audit generations, temperature 0, parse_output; sets need every variant "
                      "decided in both runs",
            "margin": "log p(hire) - log p(reject) after the forced prefix, hire when snapped "
                      "margin > 0 (grid 1/1024); op-margin-v2 score files, or version 1 where "
                      "an arm was only scored then (standard column)",
            "raw_delta": "arm - SFT, each at its own decision rule and hire rate",
            "matched_delta": "arm at margin > 0 against SFT moved to hire exactly the arm's "
                             "count (tied bin weighted, whole sets); v2 rows recomputed with "
                             "operating_point_control.matched_comparison and checked against "
                             "the published report; v1 rows are the published v1 report",
            "interval": f"{N_BOOT} bootstrap draws over candidates, seed {SEED}",
        },
        "checks": checks,
    }
    return pd.DataFrame(rows), summary


def main() -> None:
    import pyarrow as pa

    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    table, summary = build()
    out = REPO_ROOT / "reports" / "raw_vs_matched"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "table.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 220)
    cols = ["arm", "standard", "greedy_hire_rate", "greedy_unstable", "greedy_delta_pp",
            "greedy_ci_low", "greedy_ci_high", "margin_hire_rate", "margin_unstable",
            "margin_delta_pp", "matched_sft_hire_rate", "matched_sft_unstable",
            "matched_delta_pp", "matched_ci_low", "matched_ci_high", "conforms"]
    print(table[[c for c in cols if c in table.columns]].round(4).to_string(index=False))
    print(json.dumps(summary["checks"], indent=1))


if __name__ == "__main__":
    main()
