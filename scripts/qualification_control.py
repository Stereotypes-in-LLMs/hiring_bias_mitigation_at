#!/usr/bin/env python
"""Reads the identity x qualification test (docs/QUALIFICATION_CONTROL.md) from the op-margin-v2
margin passes on the RJDB dev pairs. CPU only; every number comes from the margin files and the
pairs parquet. Arms whose pass does not exist yet are skipped, so the script can be run after
each job.

    source scripts/cscs/env.sh
    OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/qualification_control.py
    # -> reports/qualification_control/{summary,direction,curves,bands,by_qualification}.csv, calls.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import operating_point_control as opc  # noqa: E402

from hiring_bias_mitigation.eval import runner as R  # noqa: E402
from hiring_bias_mitigation.utils.config import load_config, resolve_output_path  # noqa: E402

_spec = importlib.util.spec_from_file_location("final_table", REPO / "scripts/final_table.py")
FT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FT)

QUAL = REPO / "configs/audit/qual"
SFT_CFG = QUAL / "qwen3.5-9b_en_sft_v3_additional_rjdbdev.yaml"
RECIPES = {   # recipe: (config stem per seed 42/43/44)
    "DPO_V3_4880": ["qwen3.5-9b_en_dpo_v3_4880_step305_rjdbdev",
                    "qwen3.5-9b_en_dpo_v3_4880_seed43_step305_rjdbdev",
                    "qwen3.5-9b_en_dpo_v3_4880_seed44_step305_rjdbdev"],
    "GRPO+decay R4": ["qwen3.5-9b_en_grpo_improve_r4_tipped_rjdbdev",
                      "qwen3.5-9b_en_grpo_improve_r4_tipped_seed43_rjdbdev",
                      "qwen3.5-9b_en_grpo_improve_r4_tipped_seed44_rjdbdev"],
    "Distill": ["qwen3.5-9b_en_decision_distill_seed42_rjdbdev",
                "qwen3.5-9b_en_decision_distill_seed43_rjdbdev",
                "qwen3.5-9b_en_decision_distill_seed44_rjdbdev"],
}
SEEDS = (42, 43, 44)
PAIRS = "artifacts/qualification_control/pairs_rjdb_dev.parquet"
N_BOOT, BOOT_SEED = 1000, 42
TOL_PP, UNDETERMINED_PP = 2.0, 5.0      # C2/C3 tolerance; half-width that reads "undetermined"
OUT = REPO / "reports/qualification_control"


def run_name(stem: str) -> str:
    return R.build_run_name(load_config(QUAL / f"{stem}.yaml"))


def load(run: str, which: str) -> pd.DataFrame | None:
    try:
        return opc.load_scores(run, which)[0]
    except SystemExit:
        return None


def triple_of(pair_id: pd.Series) -> pd.Series:
    return pair_id.str.rsplit("__", n=1).str[0]


def quality_of(pair_id: pd.Series) -> pd.Series:
    return pair_id.str.rsplit("__", n=1).str[1]


def direction(frame: pd.DataFrame, attr_free_only: bool = True) -> pd.Series:
    """Per triple: whether margin(matched) > margin(weakened), averaged over the triple's
    variants (the attribute-free prompt alone, or the 68 cue variants)."""
    f = frame[frame["condition"] == "attr_free"] if attr_free_only else frame[frame["condition"] != "attr_free"]
    f = f.assign(triple=triple_of(f["pair_id"]), quality=quality_of(f["pair_id"]))
    wide = f.pivot_table(index=["triple", "protected_group", "protected_attr", "condition"],
                         columns="quality", values="margin")
    correct = (wide["matched"] > wide["weakened"]).astype(float)
    return correct.groupby(level="triple").mean()


def boot_diff(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """Paired difference of two per-triple scores (a - b), percentage points, and its candidate
    bootstrap interval (one triple is one candidate)."""
    a, b = a.align(b, join="inner")
    d = (a - b).to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = 100 * d[idx].mean(axis=1)
    return 100 * float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def separation_at(frame: pd.DataFrame, k: int) -> float:
    """M3: with the arm cut to hire `k` prompts, the share of triples whose matched CV is hired
    and weakened CV rejected, minus the reverse, on the attribute-free prompts (tie bins
    weighted as matched_flags weights them)."""
    m = opc.snap(frame["margin"])
    level, w = opc.cut_for_count(m, k)
    hire = pd.Series(np.where(m > level, 1.0, np.where(m == level, w, 0.0)), index=frame.index)
    f = frame[frame["condition"] == "attr_free"].assign(hire=hire, triple=triple_of(frame["pair_id"]),
                                                          quality=quality_of(frame["pair_id"]))
    wide = f.pivot_table(index="triple", columns="quality", values="hire")
    sep = wide["matched"] * (1 - wide["weakened"]) - wide["weakened"] * (1 - wide["matched"])
    return 100 * float(sep.mean())


def by_qualification(frame: pd.DataFrame, k: int) -> dict:
    """M4: unstable share at the common cut on the matched and on the weakened pairs, and the
    median spread of a set's cue variants on each."""
    flags, _ = opc.matched_flags(frame, "margin", k)
    quality = quality_of(pd.Series(flags.index.get_level_values("pair_id")))
    att = frame[frame["condition"] != "attr_free"].assign(m=opc.snap(frame.loc[frame["condition"] != "attr_free", "margin"]))
    g = att.groupby(opc.SET)["m"]
    spread = (g.max() - g.min())
    sq = quality_of(pd.Series(spread.index.get_level_values("pair_id")))
    out = {}
    for q in ("matched", "weakened"):
        out[f"unstable_{q}_pct"] = 100 * float(flags.to_numpy()[quality.to_numpy() == q].mean())
        out[f"spread_{q}_median"] = float(np.median(spread.to_numpy()[sq.to_numpy() == q]))
    return out


def main() -> None:
    pairs = pd.read_parquet(resolve_output_path(PAIRS))
    sft_run = run_name(SFT_CFG.stem)
    sft, base = load(sft_run, "adapter"), load(sft_run, "base")
    if sft is None:
        raise SystemExit("no SFT-v3 margin pass on the RJDB dev pairs yet")
    n = len(sft)
    k_common = int((opc.snap(sft["margin"]) > 0).sum())
    arms = [("Base", base), ("SFT-v3", sft)]
    for recipe, stems in RECIPES.items():
        for seed, stem in zip(SEEDS, stems, strict=True):
            arms.append((f"{recipe} s{seed}", load(run_name(stem), "adapter")))
    arms = [(label, f) for label, f in arms if f is not None]
    present = {label for label, _ in arms}
    print("passes present:", ", ".join(present))

    dir_free = {label: direction(f, True) for label, f in arms}
    dir_all = {label: direction(f, False) for label, f in arms}
    rows, drows, curves = [], [], []
    for label, f in arms:
        hire, unstable = opc.unstable_share(f, "margin", 0.0)
        row = {"label": label, "hire_rate_greedy_pct": 100 * hire, "unstable_greedy_pct": 100 * unstable,
               "common_hire_rate_pct": 100 * k_common / n,
               "unstable_common_pct": 100 * opc.matched_flags(f, "margin", k_common)[1]["unstable"],
               "direction_free_pct": 100 * float(dir_free[label].mean()),
               "direction_variants_pct": 100 * float(dir_all[label].mean()),
               "separation_common_pct": separation_at(f, k_common),
               **by_qualification(f, k_common)}
        if label == "SFT-v3" and base is not None:
            c = opc.matched_comparison(base, sft)              # the base moved to SFT-v3's rate
            row.update({"vs": "Base", "delta_own_pp": c["delta_matched_pp"], "ci_own_low": c["ci_low"],
                        "ci_own_high": c["ci_high"], "base_at_sft_rate_pct": 100 * c["unstable_ref_matched"]})
        elif label not in ("Base", "SFT-v3"):
            own = opc.matched_comparison(sft, f)                # SFT-v3 moved to the arm
            common = opc.matched_comparison(f, sft)             # the arm moved to SFT-v3
            row.update({"vs": "SFT-v3", "delta_own_pp": own["delta_matched_pp"],
                        "ci_own_low": own["ci_low"], "ci_own_high": own["ci_high"],
                        "delta_common_pp": -common["delta_matched_pp"],
                        "ci_common_low": -common["ci_high"], "ci_common_high": -common["ci_low"]})
        rows.append(row)
        ref_label = "Base" if label == "SFT-v3" else "SFT-v3"
        if label != "Base" and ref_label in dir_free:
            for name, store in (("attr_free", dir_free), ("variants", dir_all)):
                d, lo, hi = boot_diff(store[label], store[ref_label])
                drows.append({"label": label, "vs": ref_label, "variants": name,
                              "direction_pct": 100 * float(store[label].mean()),
                              "ref_direction_pct": 100 * float(store[ref_label].mean()),
                              "diff_pp": d, "ci_low": lo, "ci_high": hi})
        curves += [{"label": label, "hire_rate_pct": 100 * h, "unstable_pct": u}
                   for h, u in zip(FT.HIRE_GRID, FT.curve(f))]
    # Post hoc, descriptive (not registered): on these pairs the models hire about 37%, so the
    # registered 10-35% grid lies below their own operating points, unlike the development
    # benchmark where 35% was 12.5 pp above. The same curve continued to 50%, and the mean of
    # (arm - SFT-v3) over 37-50%, say what happens above the operating point here.
    ext_grid = np.round(np.arange(0.35, 0.5001, 0.005), 3)
    ext = {label: np.array(FT.curve(f, ext_grid)) for label, f in arms}
    above = []
    for label, _ in arms:
        if label != "SFT-v3":
            diff = ext[label] - ext["SFT-v3"]
            above.append({"label": label, "band_37_50_pp": float(diff[ext_grid >= 0.37].mean()),
                          "at_50_pp": float(diff[-1])})
    pd.DataFrame(above).to_csv(OUT / "bands_above_operating_point.csv", index=False)
    summary, directions, curve_table = pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(curves)
    curve_table["run"] = curve_table["label"]
    FT.REFERENCE = "SFT-v3"                                   # curve_bands keys on the run column
    bands = FT.curve_bands(curve_table) if len(arms) > 1 else pd.DataFrame()

    # The registered calls.
    calls = {}
    s = summary.set_index("label")
    if "Base" in present:
        calls["C1_replication"] = ("SFT-v3 better than Base" if s.loc["SFT-v3", "ci_own_high"] < 0
                                   else "does not replicate")
        d = directions[(directions["label"] == "SFT-v3") & (directions["variants"] == "attr_free")].iloc[0]
        half = (d["ci_high"] - d["ci_low"]) / 2
        calls["C2_selectivity"] = ("undetermined (interval wider than ±5 pp)" if half > UNDETERMINED_PP
                                   else "keeps qualification sensitivity" if d["diff_pp"] >= -TOL_PP
                                   else "SFT also dulls qualification sensitivity")
        calls["C2_diff_pp"] = [round(d["diff_pp"], 2), round(d["ci_low"], 2), round(d["ci_high"], 2)]
    for recipe in RECIPES:
        labels = [f"{recipe} s{seed}" for seed in SEEDS]
        if all(lab in present for lab in labels):
            below = all(s.loc[lab, "ci_common_high"] < 0 for lab in labels)
            mean_dir = float(np.mean([s.loc[lab, "direction_free_pct"] for lab in labels]))
            keeps = mean_dir >= s.loc["SFT-v3", "direction_free_pct"] - TOL_PP
            calls[f"C3_{recipe}"] = "better than SFT-v3 on these CVs" if (below and keeps) else "not better"
    if len(bands):
        b = bands.set_index("label")
        for recipe, bar in (("DPO_V3_4880", None), ("GRPO+decay R4", 1.0), ("Distill", 1.0)):
            labels = [f"{recipe} s{seed}" for seed in SEEDS if f"{recipe} s{seed}" in b.index]
            if len(labels) == 3:
                band = float(np.mean([b.loc[lab, "band_30_35_pp"] for lab in labels]))
                ok = band > 0 if bar is None else abs(band) <= bar
                calls[f"C4_{recipe}_band_30_35_pp"] = [round(band, 2), "as predicted" if ok else "not as predicted"]
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "summary.csv", index=False)
    directions.to_csv(OUT / "direction.csv", index=False)
    curve_table.drop(columns="run").to_csv(OUT / "curves.csv", index=False)
    bands.to_csv(OUT / "bands.csv", index=False)
    (OUT / "calls.json").write_text(json.dumps(calls, indent=2))
    pd.set_option("display.width", 250)
    print(summary.round(2).to_string(index=False))
    print(directions.round(2).to_string(index=False))
    if len(bands):
        print(bands.round(2).to_string(index=False))
    print(pd.DataFrame(above).round(2).to_string(index=False))
    print(json.dumps(calls, indent=2))


if __name__ == "__main__":
    main()
