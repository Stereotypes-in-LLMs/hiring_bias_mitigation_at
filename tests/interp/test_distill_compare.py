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


def test_pair_sq_matches_the_pairwise_sum():
    v = np.array([[1.0], [3.0], [6.0], [2.0], [5.0]])
    groups = np.array(["a", "a", "a", "b", "b"])
    total, count = DC._pair_sq(v, groups)
    assert count == 4 and abs(total - ((1 - 3) ** 2 + (1 - 6) ** 2 + (3 - 6) ** 2 + 9)) < 1e-9


def test_noise_biases_the_raw_ratio_up_and_the_correction_removes_it():
    """A pure rescaling by 0.5 seen through additive noise: the smaller model's |dm| is inflated
    more, so the raw rho exceeds 1; subtracting the known noise variance brings it back."""
    manifest, m_ref, m_spec = toy(0.5, 1.0, n=4000, seed=1)
    rng = np.random.default_rng(2)
    var = 0.15 ** 2
    manifest["dm_ref"] += rng.normal(0, np.sqrt(var), len(manifest))
    manifest["dm_spec"] += rng.normal(0, np.sqrt(var), len(manifest))
    noise = {"ref": {"var_dm": var}, "spec": {"var_dm": var}}
    out = DC.sensitivity(manifest, m_ref, m_spec, "spec", "ref", ["x"], n_boot=20, seed=0,
                         noise=noise).iloc[0]
    assert out["rho_rms"] > 1.0
    assert abs(out["rho_noise_corrected"] - 1.0) < 0.03
