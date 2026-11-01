"""Margin readout of each audited arm against a reference arm, from the standard's own margins
(docs/PATCHING_DISTILL.md, section 4; post hoc, deciding nothing).

    python scripts/margin_readout.py

For every row of PAIRS, on the prompts both arms scored (op-margin-v2 files under
$HBM_OUTPUT_ROOT/outputs/operating_point/v2/):
  a_median          the global margin scale: median attribute-free |m|, arm over reference
                    (scale_only_control.scale_factors, unchanged);
  spread_ratio      the attribute deviations: median over attributed sets of the margins'
                    standard deviation across the set's variants, arm over reference;
  rho               spread_ratio / a_median: below 1 the attribute deviations shrink more than
                    the scale, above 1 less;
  border_growth     median attribute-free |m| on SFT-v3's 35 borderline pairs (attribute-free
                    |m| < 2, scale_only_control.borderline_pairs), arm over reference, and
  border_over_global  border_growth / a_median: how much further the arm moves those pairs from
                    the cut than everything else;
  fixed_overlap_d42 for an arm read against SFT-v3, the weighted Jaccard of the sets it fixes
                    with those all-layer distillation s42 fixes (scale_only_control.set_flags:
                    a fixed set weighs SFT-v3's matched flag times one minus the arm's).
Margins near the cut sit on a 1/8-logit grid (scale_only_control.RAW_GRID), so ratios of small
medians are coarse. Writes reports/margin_readout/readout.csv. CPU only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "scale_only_control", ROOT / "scripts" / "scale_only_control.py")
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)
OPC = SC.OPC

SFT_V3 = SC.REF
RUN = "Qwen3.5-9B--en--{}--adapter--en_only_{}"
DPO42 = RUN.format("dpo", "dpo_v3_4880_step305")
GRPO42 = RUN.format("grpo", "grpo_improve_r4_tipped_v2")
#: (label, arm run, reference run). Students are read against their parent as well.
PAIRS = [
    *[(f"distill s{s}", RUN.format("sft", f"decision_distill_seed{s}"), SFT_V3)
      for s in (42, 43, 44)],
    ("DPO_V3_4880 s42", DPO42, SFT_V3),
    *[(f"DPO_V3_4880 s{s}", RUN.format("dpo", f"dpo_v3_4880_seed{s}_step305"), SFT_V3)
      for s in (43, 44)],
    ("GRPO+decay R4 s42", GRPO42, SFT_V3),
    *[(f"GRPO+decay R4 s{s}", RUN.format("grpo", f"grpo_improve_r4_tipped_seed{s}"), SFT_V3)
      for s in (43, 44)],
    ("DPO student s42", RUN.format("sft", "decision_distill_dpo4880_seed42"), SFT_V3),
    ("GRPO student s42", RUN.format("sft", "decision_distill_grpo_r4_seed42"), SFT_V3),
    ("DPO student s42 / parent", RUN.format("sft", "decision_distill_dpo4880_seed42"), DPO42),
    ("GRPO student s42 / parent", RUN.format("sft", "decision_distill_grpo_r4_seed42"), GRPO42),
    *[(f"A layers 15-20 s{s}", RUN.format("sft", f"decision_distill_l15_20_seed{s}"), SFT_V3)
      for s in (42, 43, 44)],
    ("C window + GDN s42", RUN.format("sft", "decision_distill_w15_20_seed42"), SFT_V3),
    ("D rank 8 s42", RUN.format("sft", "decision_distill_r8_seed42"), SFT_V3),
    *[(f"F window removed s{s}", RUN.format("sft", f"decision_distill_seed{s}_xw15_20"), SFT_V3)
      for s in (42, 43, 44)],
    ("E floor 4 s42", RUN.format("sft", "decision_distill_floor4_seed42"), SFT_V3),
]


def set_spread(both: pd.DataFrame, column: str) -> pd.Series:
    attributed = both[both["condition"] != "attr_free"]
    return attributed.groupby(OPC.SET)[column].std(ddof=0)


def fixed(ref: pd.DataFrame, run: pd.DataFrame) -> pd.Series:
    flags = SC.set_flags(ref, run)
    return flags["ref"] * (1 - flags["run"])


def overlap(a: pd.Series, b: pd.Series) -> float:
    a, b = a.align(b, fill_value=0.0)
    high = float(np.maximum(a, b).sum())
    return float(np.minimum(a, b).sum()) / high if high else float("nan")


def readout(ref: pd.DataFrame, run: pd.DataFrame, border: set[str]) -> dict:
    scale = SC.scale_factors(ref, run)
    both = OPC._paired(ref, run)
    spread = (set_spread(both, "margin_run").median()
              / set_spread(both, "margin_ref").median())
    free = both[(both["condition"] == "attr_free") & both["pair_id"].isin(list(border))]
    growth = free["margin_run"].abs().median() / free["margin_ref"].abs().median()
    return {"prompts": len(both), "a_median": scale["a_median"], "spread_ratio": float(spread),
            "rho": float(spread / scale["a_median"]), "border_growth": float(growth),
            "border_over_global": float(growth / scale["a_median"])}


def main() -> None:
    out = ROOT / "reports" / "margin_readout"
    out.mkdir(parents=True, exist_ok=True)
    border = SC.borderline_pairs(OPC.load_scores(SFT_V3, "adapter")[0])
    cache: dict[str, pd.DataFrame] = {}

    def scores(run: str) -> pd.DataFrame | None:
        if run not in cache:
            try:
                cache[run] = OPC.load_scores(run, "adapter")[0]
            except SystemExit:  # load_scores refuses a run with no op-margin-v2 file
                cache[run] = None
        return cache[run]

    d42 = fixed(scores(SFT_V3), scores(PAIRS[0][1]))
    rows = []
    for label, run, ref in PAIRS:
        a, b = scores(run), scores(ref)
        if a is None or b is None:
            print(f"skipped {label}: no op-margin-v2 file yet")
            continue
        extra = {"fixed_overlap_d42": overlap(fixed(b, a), d42)} if ref == SFT_V3 else {}
        rows.append({"label": label, "run": run, "reference": ref, **readout(b, a, border),
                     **extra})
    table = pd.DataFrame(rows)
    table.to_csv(out / "readout.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{len(border)} borderline pairs\n"
          + table.drop(columns=["run", "reference"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
