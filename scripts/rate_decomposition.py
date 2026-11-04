#!/usr/bin/env python
"""Where a post-SFT arm's unstable sets come from at a given hire rate.

At one hire rate the curve (reports/final/curves.csv) gives one number per arm. This script
splits it, against SFT-v3 cut to the same hire count: the within-group sets both leave unstable,
the sets the arm fixes (unstable for SFT-v3, stable for the arm) and the sets it breaks (the
reverse), tie bins weighted as `matched_flags` weights them. For the broken sets it reads how the
arm did it: the spread of the set's attributed variants (max - min of the snapped margins) and
the distance of the set's mean margin from the cut, for SFT-v3 and for the arm. Also where the
broken weight sits, by cue group and wording, and on SFT-v3's borderline pairs. Descriptive and
CPU only, from the op-margin-v2 margins; a reviewer's question on the poster's 35% point
(2026-11-04).

    source scripts/cscs/env.sh
    OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/rate_decomposition.py
    # -> reports/final/rate_decomposition.csv, rate_decomposition_cells.csv, rate_decomposition_ranks.csv
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import operating_point_control as opc  # noqa: E402

_spec = importlib.util.spec_from_file_location("seed_overlap", REPO / "scripts/seed_overlap.py")
SO = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SO)
SC, REF, RECIPES, SEEDS = SO.SC, SO.SC.REF, SO.RECIPES, SO.SEEDS
RATES = {"common": None, "35": 0.35}   # None: SFT-v3's own greedy hire count


def set_geometry(both: pd.DataFrame, column: str, level: float) -> pd.DataFrame:
    """Per set: spread of the attributed variants' snapped margins and the set mean's distance
    from the cut `level`."""
    att = both[both["condition"] != "attr_free"].assign(m=opc.snap(both.loc[
        both["condition"] != "attr_free", column]))
    g = att.groupby(opc.SET)["m"]
    return pd.DataFrame({"spread": g.max() - g.min(), "to_cut": (g.mean() - level).abs()})


def rank_geometry(both: pd.DataFrame, column: str, hire_rate: float) -> pd.DataFrame:
    """Per set, on the rank scale: each prompt's margin as its percentile among all the arm's
    prompts (ties averaged), the set's rank interval over its attributed variants (max - min,
    in pp of prompts) and the distance of the interval's centre from the cut percentile
    (100 - hire rate). A monotone change of every margin leaves this unchanged, so it reads
    the selection at a fixed hire rate rather than the logit scale."""
    u = pd.Series(100 * rankdata(opc.snap(both[column])) / len(both), index=both.index)
    att = both[both["condition"] != "attr_free"].assign(u=u)
    g = att.groupby(opc.SET)["u"]
    return pd.DataFrame({"rank_width": g.max() - g.min(),
                         "rank_to_cut": ((g.max() + g.min()) / 2 - (100 - 100 * hire_rate)).abs()})


def wmedian(values: pd.Series, weights: pd.Series) -> float:
    v, w = values.to_numpy(float), weights.to_numpy(float)
    if w.sum() == 0:
        return float("nan")
    order = np.argsort(v)
    cum = np.cumsum(w[order])
    return float(v[order][np.searchsorted(cum, cum[-1] / 2)])


def main() -> None:
    ref = opc.load_scores(REF, "adapter")[0]
    border = SC.borderline_pairs(ref)
    k_common = int((opc.snap(ref["margin"]) > 0).sum())
    rows, cells, ranks = [], [], []
    for recipe, runs in RECIPES.items():
        for seed, run in zip(SEEDS, runs, strict=True):
            both = opc._paired(ref, opc.load_scores(run, "adapter")[0])
            n = len(both)
            for name, rate in RATES.items():
                k = k_common if rate is None else round(rate * n)
                fr, ir = opc.matched_flags(both, "margin_ref", k)
                fu, iu = opc.matched_flags(both, "margin_run", k)
                shared = np.minimum(fr, fu)
                fixed, broken = fr - shared, fu - shared
                geo_r = set_geometry(both, "margin_ref", ir["threshold"])
                geo_u = set_geometry(both, "margin_run", iu["threshold"])
                all_w = pd.Series(1.0, index=fr.index)
                on_border = fr.index.get_level_values("pair_id").isin(list(border))
                row = {"recipe": recipe, "seed": seed, "rate": name,
                       "hire_rate_pct": 100 * k / n, "sets": len(fr),
                       "sft_unstable": float(fr.sum()), "arm_unstable": float(fu.sum()),
                       "shared": float(shared.sum()), "fixed": float(fixed.sum()),
                       "broken": float(broken.sum()),
                       "broken_on_borderline_share": SC.share(broken, on_border),
                       "fixed_on_borderline_share": SC.share(fixed, on_border),
                       "spread_sft_broken": wmedian(geo_r["spread"], broken),
                       "spread_arm_broken": wmedian(geo_u["spread"], broken),
                       "spread_sft_all": wmedian(geo_r["spread"], all_w),
                       "spread_arm_all": wmedian(geo_u["spread"], all_w),
                       "to_cut_sft_broken": wmedian(geo_r["to_cut"], broken),
                       "to_cut_arm_broken": wmedian(geo_u["to_cut"], broken),
                       "to_cut_sft_all": wmedian(geo_r["to_cut"], all_w),
                       "to_cut_arm_all": wmedian(geo_u["to_cut"], all_w)}
                rows.append(row)
                # Rank-scale geometry of the same sets, for the four groups the two arms define.
                rk_r, rk_u = rank_geometry(both, "margin_ref", k / n), rank_geometry(both, "margin_run", k / n)
                neither = (1 - np.maximum(fr, fu))
                for group, weight in (("broken", broken), ("fixed", fixed), ("shared", shared), ("neither", neither)):
                    for arm, rk in (("sft", rk_r), (recipe, rk_u)):
                        ranks.append({"recipe": recipe, "seed": seed, "rate": name, "group": group,
                                      "arm": arm, "weight": float(weight.sum()),
                                      "rank_width_median_pp": wmedian(rk["rank_width"], weight),
                                      "rank_to_cut_median_pp": wmedian(rk["rank_to_cut"], weight)})
                frame = pd.DataFrame({"broken": broken, "fixed": fixed, "all": all_w}).reset_index()
                for (group, cond), part in frame.groupby(["protected_group", "condition"]):
                    cells.append({"recipe": recipe, "seed": seed, "rate": name,
                                  "protected_group": group, "condition": cond,
                                  "sets": len(part), "broken": float(part["broken"].sum()),
                                  "fixed": float(part["fixed"].sum()),
                                  "broken_share": SC.share(broken, ((frame["protected_group"] == group)
                                                                    & (frame["condition"] == cond)).to_numpy()),
                                  "set_share": len(part) / len(frame)})
    out = REPO / "reports/final"
    table, cell_table, rank_table = pd.DataFrame(rows), pd.DataFrame(cells), pd.DataFrame(ranks)
    table.to_csv(out / "rate_decomposition.csv", index=False)
    cell_table.to_csv(out / "rate_decomposition_cells.csv", index=False)
    rank_table.to_csv(out / "rate_decomposition_ranks.csv", index=False)
    print(rank_table[rank_table["recipe"] == "DPO_V3_4880"].pivot_table(
        index=["rate", "group", "arm"], columns="seed",
        values=["weight", "rank_width_median_pp", "rank_to_cut_median_pp"]).round(2).to_string())
    pd.set_option("display.width", 250)
    print(table.round(2).to_string(index=False))
    print(cell_table[cell_table["rate"] == "35"].groupby(["recipe", "protected_group", "condition"])
          [["broken_share", "set_share"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
