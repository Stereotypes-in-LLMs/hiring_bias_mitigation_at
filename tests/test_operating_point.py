"""Operating-point control: comparing two fine-tuned arms at the same hire rate."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    assert abs(result["hire_rate_ref_matched"] - result["hire_rate_run"]) < 1e-12
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


def _sets(margins: list[list[float]], **flags: list[list[bool]]) -> pd.DataFrame:
    """One counterfactual set per inner list, one prompt per margin; `flags` per prompt."""
    rows = []
    for s, values in enumerate(margins):
        for a, margin in enumerate(values):
            rows.append({"pair_id": f"c{s}__j{s}", "protected_group": "g",
                         "protected_attr": f"a{a}", "condition": "explicit", "margin": margin,
                         **{name: flag[s][a] for name, flag in flags.items()}})
    return pd.DataFrame(rows)


def _run_hiring(k: int, shape: list[list[float]]) -> pd.DataFrame:
    """A compared arm with the same prompts as `shape` that hires its first `k` prompts."""
    n = sum(len(v) for v in shape)
    hires = iter([1.0] * k + [-1.0] * (n - k))
    return _sets([[next(hires) for _ in v] for v in shape])


def test_the_cut_hires_exactly_k_prompts_with_a_weight_on_the_tied_bin():
    m = np.array([1.0] * 3 + [0.0] * 4 + [-1.0] * 3)
    assert OP.cut_for_count(m, 5) == (0.0, 0.5)    # 3 above, half of the 4 tied at 0
    assert OP.cut_for_count(m, 3) == (0.0, 0.0)    # a bin edge: no weight needed
    assert OP.cut_for_count(m, 0) == (1.0, 0.0)
    assert OP.cut_for_count(m, 10) == (-1.0, 1.0)


def test_a_tied_bin_is_matched_exactly_and_kept_whole_inside_each_set():
    # 20 prompts on a 1/8 grid, 12 of them tied at 0.0. A run hiring 8 needs 3 of those 12;
    # `margin > quantile` hires 5 prompts in all (25% instead of 40%).
    ref = _sets([[0.0] * 5,                      # all tied: no cut can make this set unstable
                 [0.125, 0.0, 0.0, 0.0, 0.0],      # unstable unless its ties are hired
                 [0.0, 0.0, -0.125, -0.125, -0.125],  # unstable only if they are
                 [0.25, 0.25, 0.25, 0.125, 0.0]])  # unstable unless its tie is hired
    assert (ref["margin"] > ref["margin"].quantile(1 - 0.4)).mean() == 0.25
    run = _run_hiring(8, [[0] * 5] * 4)
    result = OP.matched_comparison(ref, run, n_boot=50)
    assert result["hire_rate_run"] == 0.4
    assert abs(result["hire_rate_ref_matched"] - 0.4) < 1e-12
    assert (result["ref_threshold_matched"], result["ref_tied_prompts"]) == (0.0, 12)
    assert result["ref_tie_weight"] == 3 / 12
    # Without the bin sets 2 and 4 are unstable, with it set 3: never set 1, which splitting
    # the tied prompts one by one would make unstable too.
    assert result["unstable_ref_bin_excluded"] == 0.5
    assert result["unstable_ref_bin_included"] == 0.25
    assert abs(result["unstable_ref_matched"] - (0.75 * 0.5 + 0.25 * 0.25)) < 1e-12
    assert result["floor_check"] == "unverified"   # these frames carry no floor flags
    assert result["floor_bound_pp"] is None


def test_ties_inside_a_set_never_make_it_unstable():
    """Every set's prompts share one margin: at any hire rate the reference has no unstable
    set. Splitting the tied bin per prompt would invent some."""
    rng = np.random.default_rng(0)
    levels = rng.integers(-8, 8, size=60) / 8
    ref = _sets([[float(v)] * 5 for v in levels])
    shape = [[0] * 5] * 60
    for k in (37, 101, 150, 233):
        result = OP.matched_comparison(ref, _run_hiring(k, shape), n_boot=20)
        assert abs(result["hire_rate_ref_matched"] - k / 300) < 1e-12
        assert result["unstable_ref_matched"] == 0.0


def test_float_noise_does_not_split_a_set():
    """vLLM's margins carry float noise of ~1e-7 on a coarse grid. A quantile lands on a noisy
    value and `margin > t` splits sets whose prompts sit on the same grid value."""
    rng = np.random.default_rng(1)
    levels = np.repeat([1.5, 1.375, 1.25], [10, 20, 10])
    noisy = [[float(v + rng.normal(0, 1e-7)) for _ in range(5)] for v in levels]
    ref = _sets(noisy)
    shape = [[0] * 5] * len(levels)
    run = _run_hiring(110, shape)                   # 50 above 1.375, 60 of its 100 tied prompts
    # What the quantile rule did: the threshold is noise inside the 1.375 bin and splits sets.
    t = ref["margin"].quantile(1 - 110 / 200)
    assert t != 1.375 and abs(t - 1.375) < 1e-6
    split = ref.assign(hire=ref["margin"] > t).groupby("pair_id")["hire"].nunique() > 1
    assert split.any()
    result = OP.matched_comparison(ref, run, n_boot=20)
    assert result["ref_threshold_matched"] == 1.375
    assert result["ref_tied_prompts"] == 100 and result["ref_tie_weight"] == 0.6
    assert result["unstable_ref_matched"] == 0.0
    assert abs(result["hire_rate_ref_matched"] - 0.55) < 1e-12


def test_a_cut_among_floor_substituted_margins_is_refused():
    """A saturated arm: every hire logprob fell outside the top 20 and was floored to about
    -20. Moving it to another arm's hire rate puts the cut among the floors."""
    rng = np.random.default_rng(2)
    floors = [[float(-20 - rng.integers(0, 8) / 8) for _ in range(5)] for _ in range(40)]
    ref = _sets(floors, hire_missing=[[True] * 5] * 40, reject_missing=[[False] * 5] * 40)
    shape = [[0] * 5] * 40
    run = _run_hiring(60, shape).assign(hire_missing=False, reject_missing=False)
    result = OP.matched_comparison(ref, run, n_boot=20)
    assert result["ref_threshold_matched"] < -19
    assert result["floor_check"] == "refused"
    assert result["delta_matched_pp"] is None and result["ci_low"] is None
    assert result["floor_uncertain_prompts"]["run"] == 0
    with pytest.raises(SystemExit, match="refused"):
        OP._guard_floor({"run": "x", **result}, Path("x.json"), OP.MAX_FLOOR_PP)


def test_floors_far_from_the_cut_do_not_matter():
    """An SFT-like arm: the few floored prompts sit deep in the reject tail; the cut near 0 is
    a whole FLOOR_GAP beyond them, so their decision is settled."""
    rng = np.random.default_rng(3)
    margins = [[float(rng.integers(-16, 16) / 8) for _ in range(5)] for _ in range(40)]
    missing = [[False] * 5 for _ in range(40)]
    for s in range(5):
        margins[s][0], missing[s][0] = -15.0, True
    no = [[False] * 5] * 40
    ref = _sets(margins, hire_missing=missing, reject_missing=no)
    run = _run_hiring(90, [[0] * 5] * 40).assign(hire_missing=False, reject_missing=False)
    result = OP.matched_comparison(ref, run, n_boot=20)
    assert result["floor_check"] == "ok"
    assert result["floor_uncertain_prompts"] == {"ref": 0, "run": 0}
    assert result["delta_matched_pp"] is not None


def test_floor_uncertainty_follows_the_bound_on_the_missing_word():
    frame = pd.DataFrame({"margin_x": [-3.5, -1.5, 2.0, 0.0],
                          "hire_missing_x": [True, True, False, True],
                          "reject_missing_x": [False, False, True, True]})
    # A cut at -2: hire missing at -3.5 means a true margin below -2.5, settled; at -1.5 the
    # true margin is only known to be below -0.5, either side of the cut. Reject missing at 2.0
    # means a true margin above 1.0: settled. Both missing: never settled.
    assert OP.floor_uncertain(frame, "x", -2.0).tolist() == [False, True, False, True]
    assert OP.floor_uncertain(frame, "x", 1.5).tolist() == [False, False, True, True]
    assert OP.floor_uncertain(frame.drop(columns="hire_missing_x"), "x", 0.0) is None
