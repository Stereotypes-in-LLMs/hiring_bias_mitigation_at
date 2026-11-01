"""scripts/margin_readout.py: a pure rescaling reads as rho = 1 with its borderline pairs moved
like everything else, and shrinking only the attribute deviations reads as rho < 1."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "margin_readout", ROOT / "scripts" / "margin_readout.py")
MR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(MR)


def frame(seed: int = 0, n_pairs: int = 60, variants: int = 5) -> pd.DataFrame:
    """One attribute-free prompt per pair and a set of attributed variants around its mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        pair, mean = f"c{i}__j{i}", rng.normal(-1.0, 4.0)
        rows.append({"pair_id": pair, "protected_group": "none", "protected_attr": "none",
                     "condition": "attr_free", "margin": mean})
        for v in range(variants):
            rows.append({"pair_id": pair, "protected_group": "g", "protected_attr": f"a{v}",
                         "condition": "explicit", "margin": mean + rng.normal(0, 0.6)})
    return pd.DataFrame(rows)


def test_a_pure_rescaling_reads_as_rho_one():
    ref = frame(seed=1)
    run = ref.assign(margin=0.5 * ref["margin"])
    border = MR.SC.borderline_pairs(ref)
    out = MR.readout(ref, run, border)
    assert out["prompts"] == len(ref)
    for key in ("a_median", "spread_ratio", "border_growth"):
        assert abs(out[key] - 0.5) < 1e-9, key
    assert abs(out["rho"] - 1) < 1e-9 and abs(out["border_over_global"] - 1) < 1e-9


def test_shrinking_only_the_attribute_deviations_reads_as_rho_below_one():
    ref = frame(seed=2)
    free = ref[ref["condition"] == "attr_free"].set_index("pair_id")["margin"]
    centre = ref["pair_id"].map(free)
    run = ref.assign(margin=centre + 0.25 * (ref["margin"] - centre))  # free prompts unchanged
    out = MR.readout(ref, run, MR.SC.borderline_pairs(ref))
    assert abs(out["a_median"] - 1) < 1e-9
    assert out["rho"] < 0.5
