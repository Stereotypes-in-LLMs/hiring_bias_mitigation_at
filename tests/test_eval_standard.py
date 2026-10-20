"""The comparison standard of docs/EVAL_STANDARD.md: which arm counts as better, and why not."""

from hiring_bias_mitigation.analysis import eval_standard as S

SFT = {"run": "sft", "family": "sft", "tpr_vs_gpt4o_pct": 65.83, "agree_gpt4o_pct": 82.55}


def _arm(**overrides):
    row = {"run": "arm", "family": "dpo", "primary_against": "sft", "primary_delta_pp": -2.0,
           "primary_ci_low_pp": -3.0, "primary_ci_high_pp": -1.0, "primary_conforms": True,
           "merge_kept_pct": 91.0, "parse_failure_pct": 0.0, "refusal_pct": 0.0,
           "tpr_vs_gpt4o_pct": 66.48, "agree_gpt4o_pct": 82.61}
    row.update(overrides)
    return row


def test_a_baseline_is_not_judged():
    out = S.judge({"run": "base", "family": "none"}, None)
    assert out["std_verdict"] == "baseline" and out["guard_merge"] == S.NOT_APPLICABLE


def test_better_needs_the_interval_below_zero_and_every_guardrail():
    out = S.judge(_arm(), SFT)
    assert out["std_decision"] == "better" and out["std_verdict"] == "better"
    assert all(out[g] == S.PASS for g in ("guard_merge", "guard_tpr", "guard_agree",
                                          "guard_decided"))


def test_a_gain_within_one_point_needs_a_second_seed():
    out = S.judge(_arm(primary_delta_pp=-0.6, primary_ci_high_pp=-0.1), SFT)
    assert out["std_verdict"] == "better, pending a second seed"


def test_the_grpo_pilot_fails_merge_retention_and_tpr():
    grpo = _arm(primary_delta_pp=-0.95, primary_ci_low_pp=-1.90, primary_ci_high_pp=-0.05,
                merge_kept_pct=58.0, tpr_vs_gpt4o_pct=50.57, agree_gpt4o_pct=80.43)
    out = S.judge(grpo, SFT)
    assert out["std_decision"] == "better"
    assert out["guard_merge"] == S.FAIL and out["guard_tpr"] == S.FAIL
    assert out["guard_agree"] == S.PASS  # -2.12 pp is within 5
    assert out["std_verdict"] == "not better: fails merge, tpr"


def test_r2em_100_is_not_better_because_its_interval_reaches_zero():
    r2em = _arm(primary_delta_pp=-0.96, primary_ci_low_pp=-1.96, primary_ci_high_pp=0.04)
    out = S.judge(r2em, SFT)
    assert out["std_verdict"] == "not better: the interval reaches 0"


def test_unmeasured_guardrails_leave_a_gain_undecided():
    out = S.judge(_arm(merge_kept_pct=None), SFT)
    assert out["guard_merge"] == S.NOT_MEASURED
    assert out["std_verdict"] == "undecided: not measured merge"
    assert S.judge(_arm(), None)["guard_tpr"] == S.NO_REFERENCE


def test_a_comparison_that_does_not_conform_decides_nothing():
    out = S.judge(_arm(primary_conforms=False), SFT)
    assert out["std_verdict"] == "undecided: comparison does not conform"


def test_an_adapter_against_its_base_is_judged_against_the_baseline_row():
    sft = {**SFT, "primary_against": "base", "baseline_run": "baseline",
           "primary_delta_pp": -9.2, "primary_ci_high_pp": None, "primary_conforms": True}
    base = {"run": "baseline", "family": "none", "tpr_vs_gpt4o_pct": 41.73,
            "agree_gpt4o_pct": 78.83}
    out = S.judge_rows([sft, base])[0]
    assert out["std_reference_arm"] == "baseline"
    assert out["std_tpr_vs_ref_pp"] == 65.83 - 41.73
    assert out["std_verdict"] == "undecided: no interval"


def test_missing_prompts_count_against_the_undecided_guardrail():
    out = S.judge(_arm(decided_pct=97.0), SFT)
    assert out["guard_decided"] == S.FAIL
    assert S.judge(_arm(decided_pct=99.5), SFT)["guard_decided"] == S.PASS


def test_the_declared_reference_of_the_9b_english_cell():
    sft = "Qwen3.5-9B--en--sft--adapter--en_only_cscs"
    assert S.declared_reference(sft, "Qwen/Qwen3.5-9B", "en", "gh200") == "base"
    assert S.declared_reference("Qwen3.5-9B--en--dpo--x", "Qwen/Qwen3.5-9B", "en", "gh200") == sft
    assert S.declared_reference("Qwen3.5-4B--en--sft--x", "Qwen/Qwen3.5-4B", "en",
                                "default") is None
