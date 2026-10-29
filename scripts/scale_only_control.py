"""Scale-only control (docs/SCALE_CONTROL.md): SFT-v3's own margins rescaled by a, exactly or
read through the arms' 1/8-logit grid, compared under the standard's matching, and each arm
compared with its own rescaled SFT-v3.

    python scripts/scale_only_control.py

Reads the op-margin-v2 files under $HBM_OUTPUT_ROOT/outputs/operating_point/v2/ and writes
reports/scale_control/{summary,sweep,attribution}.csv. CPU only; the matching is
operating_point_control.matched_comparison, unchanged.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "opc", ROOT / "scripts" / "operating_point_control.py")
OPC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OPC)

#: Near the threshold every arm's margins sit on this grid (differences of bf16 logits).
RAW_GRID = 1 / 8
#: SFT-v3's attribute-free |m| below this marks a borderline pair (DECISION_DISTILL.md, 5b).
BORDERLINE = 2.0
REF = "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional"
ARMS = {
    "distill42": "Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed42",
    "distill43": "Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed43",
    "distill44": "Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed44",
    "dpo4880": "Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305",
    "grpo525": "Qwen3.5-9B--en--grpo--adapter--en_only_grpo_v3_all_step525",
}
SWEEP = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def scaled(ref: pd.DataFrame, a: float, quantize: bool) -> pd.DataFrame:
    """SFT-v3's frame with every margin times `a`, optionally rounded to RAW_GRID."""
    m = a * ref["margin"].to_numpy(float)
    if quantize:
        m = np.round(m / RAW_GRID) * RAW_GRID
    return ref.assign(margin=m)


def scale_factors(ref: pd.DataFrame, run: pd.DataFrame) -> dict[str, float]:
    """The median attribute-free |m| ratio (r_scale) and the slope through the origin."""
    both = OPC._paired(ref, run)
    free = both[both["condition"] == "attr_free"]
    m_ref, m_run = both["margin_ref"].to_numpy(float), both["margin_run"].to_numpy(float)
    median = free["margin_run"].abs().median() / free["margin_ref"].abs().median()
    return {"a_median": float(median),
            "a_slope": float((m_ref * m_run).sum() / (m_ref ** 2).sum())}


def compare(ref: pd.DataFrame, run: pd.DataFrame, n_boot: int) -> dict:
    keep = ("delta_matched_pp", "ci_low", "ci_high", "hire_rate_run", "unstable_run",
            "unstable_ref_matched", "ref_tied_prompts", "ref_tie_weight")
    out = OPC.matched_comparison(ref, run, n_boot=n_boot)
    return {k: out[k] for k in keep}


def set_flags(ref: pd.DataFrame, run: pd.DataFrame) -> pd.DataFrame:
    """Per set: the run's flag at its greedy threshold and the reference's matched flag (a
    weight in [0, 1] where the cut ties), as `matched_comparison` computes them."""
    both = OPC._paired(ref, run)
    run_hire = OPC.snap(both["margin_run"]) > 0.0
    run_flags = OPC._set_flags(both, run_hire).astype(float)
    ref_flags, _ = OPC.matched_flags(both, "margin_ref", int(run_hire.sum()))
    return pd.DataFrame({"run": run_flags, "ref": ref_flags})


def within_set_rank(ref: pd.DataFrame, run: pd.DataFrame) -> pd.Series:
    """Spearman between the two arms' margins across the attributed variants of each set."""
    both = OPC._paired(ref, run)
    both = both[both["condition"] != "attr_free"]

    def rho(g):
        if g["margin_ref"].nunique() < 2 or g["margin_run"].nunique() < 2:
            return np.nan
        return spearmanr(g["margin_ref"], g["margin_run"]).statistic

    return both.groupby(OPC.SET)[["margin_ref", "margin_run"]].apply(rho)


def borderline_pairs(ref: pd.DataFrame) -> set[str]:
    free = ref[ref["condition"] == "attr_free"].groupby("pair_id")["margin"].mean().abs()
    return set(free.index[free < BORDERLINE])


def share(weights: pd.Series, mask: np.ndarray) -> float:
    return float(weights[mask].sum() / weights.sum()) if weights.sum() else np.nan


def attribution(ref: pd.DataFrame, run: pd.DataFrame, base: pd.DataFrame,
                border: set[str]) -> dict:
    """Where the arm's matched change sits: fixed and broken sets (weighted where the
    reference's cut ties), the share on borderline pairs, and within-set rank agreement with
    SFT-v3 for the arm and for its rescaled baseline."""
    f = set_flags(ref, run)
    fixed, broken = f["ref"] * (1 - f["run"]), (1 - f["ref"]) * f["run"]
    on = f.index.get_level_values("pair_id").isin(list(border))
    rank_run, rank_base = within_set_rank(ref, run), within_set_rank(ref, base)
    moved = (fixed + broken) > 0
    return {
        "sets": len(f), "borderline_pairs": len(border), "sets_on_borderline": int(on.sum()),
        "fixed": float(fixed.sum()), "broken": float(broken.sum()),
        "fixed_on_borderline_share": share(fixed, on),
        "broken_on_borderline_share": share(broken, on),
        "rank_rho_moved_sets_arm": float(rank_run[moved].mean()),
        "rank_rho_moved_sets_baseline": float(rank_base[moved].mean()),
        "rank_rho_all_sets_arm": float(rank_run.mean()),
        "rank_rho_all_sets_baseline": float(rank_base.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=OPC.N_BOOT)
    args = ap.parse_args()
    out = ROOT / "reports" / "scale_control"
    out.mkdir(parents=True, exist_ok=True)
    ref, _ = OPC.load_scores(REF, "adapter")
    border = borderline_pairs(ref)
    pd.set_option("display.width", 220)

    sweep = []
    for a in SWEEP:
        for q in (False, True):
            base = scaled(ref, a, q)
            same = bool((set_flags(ref, base)["run"] == set_flags(ref, ref)["run"]).all())
            near = int((np.abs(OPC.snap(base["margin"])) < 2).sum())
            sweep.append({"a": a, "quantized": q, "flags_identical_at_zero": same,
                          "rows_within_2": near, **compare(ref, base, args.n_boot)})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(out / "sweep.csv", index=False)
    print("sweep, SFT-v3 rescaled against SFT-v3\n" + sweep.round(4).to_string(index=False))

    rows, attr = [], []
    for name, run_name in ARMS.items():
        run, _ = OPC.load_scores(run_name, "adapter")
        a = scale_factors(ref, run)
        row = {"arm": name, **a}
        for label, value in a.items():
            for q in (False, True):
                c = compare(ref, scaled(ref, value, q), args.n_boot)
                tag = f"{'q' if q else 'exact'}_{label}"
                row[f"S_{tag}_delta"] = c["delta_matched_pp"]
                row[f"S_{tag}_ci"] = f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]"
        c = compare(ref, run, args.n_boot)
        row["arm_vs_sft_v3"] = c["delta_matched_pp"]
        row["arm_vs_sft_v3_ci"] = f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]"
        base = scaled(ref, a["a_median"], True)
        c = compare(base, run, args.n_boot)
        row["arm_vs_S_q_median"] = c["delta_matched_pp"]
        row["arm_vs_S_q_median_ci"] = f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]"
        row["rows_within_2"] = int((np.abs(OPC.snap(run["margin"])) < 2).sum())
        rows.append(row)
        attr.append({"arm": name, **attribution(ref, run, base, border)})
    summary, attr = pd.DataFrame(rows), pd.DataFrame(attr)
    summary.to_csv(out / "summary.csv", index=False)
    attr.to_csv(out / "attribution.csv", index=False)
    print("\nper arm\n" + summary.round(3).T.to_string())
    print("\nattribution\n" + attr.round(3).T.to_string())


if __name__ == "__main__":
    main()
