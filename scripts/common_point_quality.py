#!/usr/bin/env python
"""Quality against the GPT-4o reference at the common point, beside the arms' own points.

The poster's agreement with the GPT-4o reference (78.8% -> 82.4%) is read at each arm's own
hire rate, while its headline 11.8% -> 1.8% is read at the common point, 22.51% (a reviewer,
2026-11-04). This script reads agreement, TPR and FPR against the reference at both points for
the base model, SFT-v3 and every arm of reports/final/arms.csv, from the op-margin-v2 margins,
cut exactly as the final table cuts them (`matched_flags`: the bin tied at the cut hired with its
weight). Descriptive, no new measurement, CPU only.

    source scripts/cscs/env.sh
    OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/common_point_quality.py
    # -> reports/final/common_point_quality.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import operating_point_control as opc  # noqa: E402

from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

REFERENCE = "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional"
BASE_SCORES = ("Qwen3.5-9B--en--sft--adapter--en_only_cscs", "base")   # poster/base_compare.py
BASE_RAW = "Qwen3.5-9B--en--baseline--cscs"


def verdicts(raw_run: str) -> pd.DataFrame:
    raw = pd.read_parquet(resolve_output_path(f"outputs/raw/{raw_run}.parquet"),
                          columns=[*opc.KEY, "reference_decision"])
    raw["reference_decision"] = raw["reference_decision"].astype(str).str.strip().str.lower()
    return raw


N_BOOT, BOOT_SEED = 1000, 42


def per_prompt(frame: pd.DataFrame, ref: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per prompt, with the arm cut to hire exactly `k`: the expected hire flag, the reference
    verdict, agreement with it, and the candidate (the text before the first `__`)."""
    m = opc.snap(frame["margin"])
    level, w = opc.cut_for_count(m, k)
    hire = np.where(m > level, 1.0, np.where(m == level, w, 0.0))
    verdict = frame[opc.KEY].merge(ref, on=opc.KEY, how="left")["reference_decision"].to_numpy()
    out = frame[opc.KEY].copy()
    out["hire"], out["verdict"] = hire, verdict
    out["agree"] = np.where(verdict == "hire", hire, 1.0 - hire)
    out["candidate"] = out["pair_id"].str.split("__").str[0]
    return out


def quality(frame: pd.DataFrame, ref: pd.DataFrame, k: int) -> dict:
    """Expected agreement, TPR and FPR when the arm hires exactly `k` prompts."""
    p = per_prompt(frame, ref, k)
    on_hire, on_reject = p["verdict"] == "hire", p["verdict"] == "reject"
    return {"hire_pct": 100 * float(p["hire"].mean()),
            "agree_pct": 100 * float(p.loc[on_hire | on_reject, "agree"].mean()),
            "tpr_pct": 100 * float(p.loc[on_hire, "hire"].mean()),
            "fpr_pct": 100 * float(p.loc[on_reject, "hire"].mean())}


def paired_ci(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    """Paired difference (a - b) of a per-prompt column, in percentage points, with a candidate
    bootstrap interval (1,000 draws, seed 42): the same prompts on both sides, resampled by
    candidate as the standard's intervals are."""
    both = a.merge(b, on=opc.KEY, suffixes=("_a", "_b"))
    per = (both[f"{col}_a"] - both[f"{col}_b"]).groupby(both["candidate_a"]).agg(["sum", "count"])
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(per), size=(N_BOOT, len(per)))
    boots = 100 * per["sum"].to_numpy()[idx].sum(axis=1) / per["count"].to_numpy()[idx].sum(axis=1)
    return {"diff_pp": 100 * float(per["sum"].sum() / per["count"].sum()),
            "ci_low": float(np.percentile(boots, 2.5)), "ci_high": float(np.percentile(boots, 97.5))}


def main() -> None:
    ref_verdicts = verdicts(REFERENCE)
    base_verdicts = verdicts(BASE_RAW)
    same = ref_verdicts.merge(base_verdicts, on=opc.KEY, suffixes=("", "_base"))
    if not (same["reference_decision"] == same["reference_decision_base"]).all():
        raise SystemExit("the GPT-4o reference differs between the base and SFT-v3 audits")

    sft = opc.load_scores(REFERENCE, "adapter")[0]
    k_common = int((opc.snap(sft["margin"]) > 0).sum())
    arms = pd.read_csv(REPO / "reports/final/arms.csv")
    final = pd.read_csv(REPO / "reports/final/final_table.csv").set_index("run")
    models = [("Base", BASE_SCORES)] + [(a.label, (a.run, "adapter")) for a in arms.itertuples()]
    rows = []
    for label, (run, which) in models:
        try:
            frame = opc.load_scores(run, which)[0]
        except SystemExit as exc:
            print(f"skip {label}: {exc}")
            continue
        own = quality(frame, ref_verdicts, int((opc.snap(frame["margin"]) > 0).sum()))
        common = quality(frame, ref_verdicts, k_common)
        if run in final.index and run != REFERENCE:       # the same cut as the final table
            if abs(common["tpr_pct"] - float(final.loc[run, "tpr_common_pct"])) > 0.01:
                raise SystemExit(f"{label}: TPR at the common point differs from final_table.csv")
        rows.append({"label": label, "run": run, "scores": which,
                     **{f"{k}_own": v for k, v in own.items()},
                     **{f"{k}_common": v for k, v in common.items()}})
    table = pd.DataFrame(rows)
    sft_row = table[table["run"] == REFERENCE].iloc[0]
    # Paired intervals at the common point, every arm against SFT-v3: agreement over the
    # prompts with a reference verdict, TPR over those the reference hires.
    sft_pp = per_prompt(sft, ref_verdicts, k_common)
    sft_pp = sft_pp[sft_pp["verdict"].isin(["hire", "reject"])]
    ci_rows = []
    for label, (run, which) in models:
        if run == REFERENCE:
            continue
        try:
            frame = opc.load_scores(run, which)[0]
        except SystemExit:
            continue
        pp = per_prompt(frame, ref_verdicts, k_common)
        pp = pp[pp["verdict"].isin(["hire", "reject"])]
        both = pp.merge(sft_pp, on=opc.KEY, suffixes=("_a", "_b"))
        agree = paired_ci(pp, sft_pp, "agree")
        hires = pp[pp["verdict"] == "hire"]
        tpr = paired_ci(hires, sft_pp[sft_pp["verdict"] == "hire"], "hire")
        ci_rows.append({"label": label, "run": run, "prompts": len(both),
                        **{f"agree_{k}": v for k, v in agree.items()},
                        **{f"tpr_{k}": v for k, v in tpr.items()}})
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(REPO / "reports/final/common_point_quality_ci.csv", index=False)
    for col in ("agree", "tpr", "fpr"):
        table[f"{col}_common_vs_sft_pp"] = table[f"{col}_pct_common"] - sft_row[f"{col}_pct_common"]
    table.insert(3, "common_hire_pct", 100 * k_common / len(sft))
    out = REPO / "reports/final/common_point_quality.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))
    print(ci_table.round(2).to_string(index=False))
    print("-> reports/final/common_point_quality_ci.csv")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
