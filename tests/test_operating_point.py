"""Operating-point control: comparing two fine-tuned arms at the same hire rate."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "operating_point_control",
    Path(__file__).resolve().parents[1] / "scripts" / "operating_point_control.py")
OP = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(OP)


def _margins(seed: int, shift: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(40):
        level = rng.normal()
        for cond in ("explicit", "implicit"):
            for a in range(5):
                rows.append({"pair_id": f"c{c}__j{c}", "protected_group": "g",
                             "protected_attr": f"a{a}", "condition": cond,
                             "margin": level + shift + 0.3 * rng.normal()})
    return pd.DataFrame(rows)


def test_the_reference_arm_is_moved_to_the_compared_arms_hire_rate():
    ref, run = _margins(0, 0.5), _margins(1, -0.5)
    result = OP.matched_comparison(ref, run, n_boot=50)
    assert abs(result["hire_rate_ref_matched"] - result["hire_rate_run"]) <= 1 / len(run) + 1e-9
    assert result["delta_matched_pp"] == 100 * (result["unstable_run"]
                                                 - result["unstable_ref_matched"])
    assert result["ci_low"] <= result["delta_matched_pp"] <= result["ci_high"]


def test_identical_arms_differ_by_nothing():
    ref = _margins(0, 0.0)
    result = OP.matched_comparison(ref, ref.copy(), n_boot=50)
    assert result["delta_matched_pp"] == 0.0


def test_vllm_gets_the_configs_engine_kwargs():
    """On Daint the GDN prefill has to be Triton: FlashInfer's JIT needs a CUDA toolkit."""
    kwargs = OP.llm_kwargs({"engine_kwargs": {"gdn_prefill_backend": "triton"}}, "m")
    assert kwargs["gdn_prefill_backend"] == "triton" and kwargs["model"] == "m"
    assert "gdn_prefill_backend" not in OP.llm_kwargs({}, "m")
