"""scripts/scale_only_control.py (docs/SCALE_CONTROL.md): the standard's matching is invariant
to an exact rescaling of every margin, and requantizing a compression can merge a set's
variants into one tied bin."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "scale_only", ROOT / "scripts" / "scale_only_control.py")
SC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SC)


def frame(seed: int = 0, n_pairs: int = 60, variants: int = 5) -> pd.DataFrame:
    """Pairs with one attribute-free prompt and a set of attributed variants, margins on the
    1/8 grid around a pair-level mean."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        pair, mean = f"c{i}__j{i}", rng.normal(-1.0, 4.0)
        rows.append({"pair_id": pair, "protected_group": "none", "protected_attr": "none",
                     "condition": "attr_free", "margin": mean})
        for v in range(variants):
            rows.append({"pair_id": pair, "protected_group": "g", "protected_attr": f"a{v}",
                         "condition": "explicit", "margin": mean + rng.normal(0, 0.6)})
    f = pd.DataFrame(rows)
    return f.assign(margin=np.round(f["margin"] * 8) / 8)


def test_an_exact_rescaling_leaves_the_matched_change_at_zero():
    ref = frame()
    for a in (0.3, 0.46, 1.0, 2.7):
        out = SC.compare(ref, SC.scaled(ref, a, quantize=False), n_boot=50)
        assert out["delta_matched_pp"] == 0.0
        flags = SC.set_flags(ref, SC.scaled(ref, a, quantize=False))
        assert (flags["run"] == SC.set_flags(ref, ref)["run"]).all()


def test_requantizing_a_compression_merges_variants_into_ties():
    """Two variants 1/8 apart straddle 0; times 0.3 and back on the grid both sit at 0, so
    the set no longer splits at the greedy threshold."""
    ref = pd.DataFrame({"pair_id": "c0__j0", "protected_group": "g", "condition": "explicit",
                        "protected_attr": ["a0", "a1"], "margin": [0.125, -0.125]})
    exact = SC.set_flags(ref, SC.scaled(ref, 0.3, quantize=False))["run"]
    merged = SC.set_flags(ref, SC.scaled(ref, 0.3, quantize=True))["run"]
    assert exact.iloc[0] == 1.0 and merged.iloc[0] == 0.0


def test_scale_factors_recover_a_known_rescaling():
    ref = frame(seed=1)
    a = SC.scale_factors(ref, SC.scaled(ref, 0.5, quantize=False))
    assert abs(a["a_median"] - 0.5) < 1e-9 and abs(a["a_slope"] - 0.5) < 1e-9
