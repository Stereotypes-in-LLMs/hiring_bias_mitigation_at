"""scripts/interp/20_distill_compare.py: the normalised sensitivity rho (docs/DISTILL_PATCHING.md)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "distill_compare", ROOT / "scripts" / "interp" / "20_distill_compare.py")
DC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DC)


def toy(scale_all: float, scale_attr: float, n: int = 40, seed: int = 0):
    """A reference model's margins and a spec that scales every margin by `scale_all`, then
    the attribute effect again by `scale_attr`."""
    rng = np.random.default_rng(seed)
    cands = [f"c{i}" for i in range(n)]
    free_ref = rng.normal(0, 10, n)
    dm_ref = rng.normal(0, 2, n)
    free = pd.DataFrame({"candidate_id": cands, "prompt_uid": [f"f{i}" for i in range(n)],
                         "condition": "attr_free"})
    m_ref = free.assign(margin_fp32=free_ref)
    m_spec = free.assign(margin_fp32=scale_all * free_ref)
    manifest = pd.DataFrame({"contrast_id": "x", "aligned": True, "candidate_id": cands,
                             "dm_ref": dm_ref, "dm_spec": scale_all * scale_attr * dm_ref})
    return manifest, m_ref, m_spec


def test_a_global_rescaling_gives_rho_one():
    out = DC.sensitivity(*toy(0.5, 1.0), "spec", "ref", ["x"], n_boot=50, seed=0).iloc[0]
    assert abs(out["r_attr"] - 0.5) < 1e-9 and abs(out["r_scale"] - 0.5) < 1e-9
    assert abs(out["rho"] - 1.0) < 1e-9


def test_an_attribute_specific_cut_shows_below_one():
    out = DC.sensitivity(*toy(0.5, 0.6), "spec", "ref", ["x"], n_boot=200, seed=0).iloc[0]
    assert abs(out["rho"] - 0.6) < 1e-9
    assert out["rho_ci_high"] < 0.85  # the registered H1 threshold
