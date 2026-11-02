"""Overlap of the sets each training seed leaves unstable, per recipe (docs/DISTILL_FOLLOWUP.md,
section J; descriptive, deciding nothing).

    python scripts/seed_overlap.py

For each seed of a recipe, the unstable within-group sets at the seed's own greedy cut, from the
standard's op-margin-v2 margins (scale_only_control.set_flags, the `run` flag). For every pair of
seeds of one recipe: both counts, the sets unstable under both, and their Jaccard index. Beside
it, the same for SFT-v3 moved to each seed's hire rate (the matched flag, weighted where its cut
ties), which says how much overlap the operating points alone give. Also the share of each seed's
unstable sets on SFT-v3's 35 borderline pairs. Writes reports/seed_overlap/overlap.csv. CPU only.
"""

from __future__ import annotations

import importlib.util
import itertools
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

RUN = "Qwen3.5-9B--en--{}--adapter--en_only_{}"
RECIPES = {
    "DPO_V3_4880": [RUN.format("dpo", "dpo_v3_4880_step305"),
                    RUN.format("dpo", "dpo_v3_4880_seed43_step305"),
                    RUN.format("dpo", "dpo_v3_4880_seed44_step305")],
    "GRPO+decay R4": [RUN.format("grpo", "grpo_improve_r4_tipped_v2"),
                      RUN.format("grpo", "grpo_improve_r4_tipped_seed43"),
                      RUN.format("grpo", "grpo_improve_r4_tipped_seed44")],
    "Distill": [RUN.format("sft", f"decision_distill_seed{s}") for s in (42, 43, 44)],
}
SEEDS = (42, 43, 44)


def jaccard(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Weighted: the shared mass and the Jaccard index of two set weights in [0, 1]."""
    a, b = a.align(b, fill_value=0.0)
    both, either = float(np.minimum(a, b).sum()), float(np.maximum(a, b).sum())
    return both, both / either if either else float("nan")


def main() -> None:
    ref = OPC.load_scores(SC.REF, "adapter")[0]
    border = SC.borderline_pairs(ref)
    rows = []
    for recipe, runs in RECIPES.items():
        flags = {s: SC.set_flags(ref, OPC.load_scores(r, "adapter")[0])
                 for s, r in zip(SEEDS, runs, strict=True)}
        for (s, fs), (t, ft) in itertools.combinations(flags.items(), 2):
            both, jac = jaccard(fs["run"], ft["run"])
            ref_both, ref_jac = jaccard(fs["ref"], ft["ref"])
            rows.append({"recipe": recipe, "seeds": f"{s}/{t}",
                         "unstable_a": float(fs["run"].sum()), "unstable_b": float(ft["run"].sum()),
                         "unstable_both": both, "jaccard": jac,
                         "sft_v3_matched_a": float(fs["ref"].sum()),
                         "sft_v3_matched_b": float(ft["ref"].sum()),
                         "sft_v3_both": ref_both, "sft_v3_jaccard": ref_jac})
        for s, f in flags.items():
            on = f.index.get_level_values("pair_id").isin(list(border))
            rows.append({"recipe": recipe, "seeds": str(s),
                         "unstable_a": float(f["run"].sum()),
                         "on_borderline_share": SC.share(f["run"], on)})
    table = pd.DataFrame(rows)
    out = ROOT / "reports" / "seed_overlap"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "overlap.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
