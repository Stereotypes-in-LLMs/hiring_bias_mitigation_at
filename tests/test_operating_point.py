"""Operating-point control: comparing two fine-tuned arms at the same hire rate."""

import importlib.util
import json
import sys
import types
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


# ---- the standard: exact scores, provenance, refusals, conformance ------------------------

class _Logprob:
    def __init__(self, value: float):
        self.logprob = value


class _Request:
    def __init__(self, logprobs: dict, sampled: int):
        self.outputs = [types.SimpleNamespace(logprobs=[logprobs], token_ids=[sampled])]


class _LLM:
    """Answers each text from a table of next-token logprobs, keyed by token id; `drop` leaves
    some (text, token) logprobs out of the reply."""

    def __init__(self, table: dict, drop=()):
        self.table, self.drop = table, set(drop)

    def generate(self, texts, params):
        ids = params.kwargs["logprob_token_ids"]
        replies = []
        for text in texts:
            values = self.table[text]
            sampled = max(values, key=values.get)
            lp = {i: _Logprob(values[i]) for i in [*ids, sampled] if (text, i) not in self.drop}
            replies.append(_Request(lp, sampled))
        return replies


@pytest.fixture
def sampling_params(monkeypatch):
    """A stand-in `vllm` module; the list collects every SamplingParams' keyword arguments."""
    seen = []

    class SamplingParams:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SamplingParams))
    return seen


def test_the_pass_reads_both_decision_tokens_exactly_whatever_their_rank(sampling_params):
    """Version 1 asked for the top 20 and scored a word outside them at the 20th logprob - 1:
    the reject token at -31.5 below would have become a guess."""
    table = {"a": {7: -0.01, 9: -31.5, 3: -5.0}, "b": {7: -12.25, 9: -0.0001},
             "c": {7: -0.7, 9: -0.9, 3: -0.5}}
    hire, reject, decided = OP.score_logprobs(_LLM(table), ["a", "b", "c"], (7, 9), chunk=2)
    assert sampling_params[0]["logprob_token_ids"] == [7, 9]
    assert "logprobs" not in sampling_params[0]
    assert hire.tolist() == [-0.01, -12.25, -0.7]
    assert reject.tolist() == [-31.5, -0.0001, -0.9]
    assert decided == 2 / 3          # "c" continues with token 3, not a decision token


def test_a_missing_decision_logprob_stops_the_pass_instead_of_being_substituted(sampling_params):
    table = {"a": {7: -0.01, 9: -31.5}}
    with pytest.raises(RuntimeError, match="logprob_token_ids"):
        OP.score_logprobs(_LLM(table, drop={("a", 9)}), ["a"], (7, 9))


def _meta(**over) -> dict:
    meta = {"standard": OP.STANDARD, "run": "r", "which": "adapter",
            "prompts": {"n": 400, "sha1": "abc"},
            "decision": {"prefix": OP.PREFIX, "words": ["hire", "reject"], "token_ids": [7, 9],
                         "chat_template_kwargs": {"enable_thinking": False}},
            "model": {"adapters": [{"path": "a", "sha256": "1"}]},
            "software": {"vllm": "0.31.0"}, "code": {"commit": "c0ffee", "uncommitted": []}}
    return {**meta, **over}


CLEAN = {"commit": "c0ffee", "uncommitted": []}
PASSED = [{"standard": OP.STANDARD, "passed": True, "software": {"vllm": "0.31.0"}}]


def test_the_prompt_fingerprint_follows_every_prompt_key_and_the_order():
    frame = _margins(0, 0.0).head(3)
    texts = ["x", "y", "z"]
    fingerprint = OP.prompt_fingerprint(frame, texts)
    assert fingerprint == OP.prompt_fingerprint(frame.copy(), list(texts))
    assert fingerprint != OP.prompt_fingerprint(frame, ["x", "y", "w"])
    assert fingerprint != OP.prompt_fingerprint(frame.iloc[[1, 0, 2]], texts)


def test_a_score_file_is_reused_only_for_the_same_prompts_and_adapter_weights(tmp_path,
                                                                               monkeypatch):
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    path, meta_path = OP.score_paths("run", "adapter")
    plan = {"fingerprint": {"n": 400, "sha1": "abc"}, "adapters": [{"path": "a", "sha256": "1"}]}
    assert not OP.is_current(path, meta_path, plan)
    _margins(0, 0.0).to_parquet(path)
    meta_path.write_text(json.dumps(_meta()))
    assert OP.is_current(path, meta_path, plan)
    retrained = {**plan, "adapters": [{"path": "a", "sha256": "2"}]}
    assert not OP.is_current(path, meta_path, retrained)
    assert not OP.is_current(path, meta_path, {**plan, "fingerprint": {"n": 400, "sha1": "d"}})
    meta_path.write_text(json.dumps(_meta(standard="op-margin-v1")))
    assert not OP.is_current(path, meta_path, plan)


def test_comparisons_refuse_version_1_files_and_other_standards(tmp_path, monkeypatch):
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    _margins(0, 0.0).to_parquet(OP.out_dir() / "run--adapter.parquet")   # version 1
    with pytest.raises(SystemExit, match="version-1"):
        OP.load_scores("run", "adapter")
    path, meta_path = OP.score_paths("run", "adapter")
    _margins(0, 0.0).to_parquet(path)
    meta_path.write_text(json.dumps(_meta(standard="op-margin-v1")))
    with pytest.raises(SystemExit, match="op-margin-v1"):
        OP.load_scores("run", "adapter")
    meta_path.write_text(json.dumps(_meta()))
    frame, meta = OP.load_scores("run", "adapter")
    assert len(frame) == 400 and meta["standard"] == OP.STANDARD


def test_inputs_scored_on_different_prompts_or_templates_are_not_compared():
    OP.check_comparable(_meta(), _meta(run="other", software={"vllm": "0.32.0"}))
    with pytest.raises(SystemExit, match="prompts"):
        OP.check_comparable(_meta(), _meta(prompts={"n": 400, "sha1": "def"}))
    other = {**_meta()["decision"], "chat_template_kwargs": {"enable_thinking": True}}
    with pytest.raises(SystemExit, match="chat-template"):
        OP.check_comparable(_meta(), _meta(decision=other))


def test_a_report_conforms_only_on_committed_code_and_a_validated_vllm():
    assert OP.conformance([_meta(), _meta(run="sft")], CLEAN, PASSED) == []
    dirty = {"commit": "c0ffee", "uncommitted": ["scripts/operating_point_control.py"]}
    issues = OP.conformance([_meta(code=dirty), _meta(software={"vllm": "0.32.0"})], CLEAN, PASSED)
    assert any("uncommitted changes to scripts/operating_point_control.py" in i for i in issues)
    assert any("0.32.0" in i and "validate" in i for i in issues)
    assert any("different vLLM versions" in i for i in issues)
    assert any("comparison" in i for i in OP.conformance([_meta()], dirty, PASSED))
    assert OP.conformance([_meta()], CLEAN, [])                          # nothing validated yet
    assert OP.conformance([_meta()], CLEAN, [{**PASSED[0], "passed": False}])
    assert OP.conformance([_meta()], CLEAN, [{**PASSED[0], "standard": "op-margin-v1"}])


def test_an_arm_report_carries_its_inputs_method_and_conformance():
    ref, run = _margins(0, 0.5), _margins(1, -0.5)
    report = OP.arm_report("dpo", "sft", run, _meta(run="dpo"), ref, _meta(run="sft"),
                           CLEAN, PASSED)
    assert report["standard"] == OP.STANDARD and report["conforms"] and report["issues"] == []
    assert (report["inputs"]["run"]["run"], report["inputs"]["against"]["run"]) == ("dpo", "sft")
    assert report["method"]["bootstrap"] == {"draws": OP.N_BOOT, "seed": OP.BOOT_SEED,
                                             "unit": "candidate"}
    assert report["delta_matched_pp"] == OP.matched_comparison(ref, run)["delta_matched_pp"]
    unvalidated = OP.arm_report("dpo", "sft", run, _meta(run="dpo"), ref, _meta(run="sft"),
                                CLEAN, [])
    assert not unvalidated["conforms"] and unvalidated["issues"]
    with pytest.raises(SystemExit, match="prompts"):
        OP.arm_report("dpo", "sft", run, _meta(prompts={"n": 400, "sha1": "x"}), ref, _meta(),
                      CLEAN, PASSED)


def test_validation_passes_only_when_the_engines_and_the_greedy_decisions_agree():
    rng = np.random.default_rng(4)
    v = rng.normal(0, 8, size=256)
    close = v + rng.normal(0, 0.05, size=256)
    greedy = np.where(v > 0, "hire", "reject")
    ok = OP.validation_summary(v, close, greedy=greedy, v1_margin=v)
    assert ok["passed"] and ok["decision_agreement_hf"] > 0.98
    assert ok["decision_agreement_greedy"] == 1.0 and ok["v1_decision_agreement"] == 1.0
    assert not OP.validation_summary(v, rng.permutation(v))["passed"]
    assert not OP.validation_summary(v, close, greedy=np.where(v > 0, "reject", "hire"))["passed"]
    assert not OP.validation_summary(v, np.zeros_like(v))["passed"]     # constant: NaN rank corr
    sample = OP.validation_sample(31050, 256, 42)
    assert len(sample) == 256 and (np.diff(sample) > 0).all()
    assert (sample == OP.validation_sample(31050, 256, 42)).all()
