"""GRPO pilot: the prompt set, the rewards, and the weights handed to vLLM."""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from hiring_bias_mitigation.generation import grpo_prompts as G
from hiring_bias_mitigation.mitigation import grpo


def _rows(n_hire: int, n_reject: int, tipped_hire: int, tipped_reject: int) -> pd.DataFrame:
    rows = []
    for side, n, tipped in (("hire", n_hire, tipped_hire), ("reject", n_reject, tipped_reject)):
        for i in range(n):
            rows.append({"pair_id": f"{side}-p{i // 3}", "candidate_id": f"{side}-c{i // 3}",
                         "job_id": f"j{i % 5}", "lang": "en", "reference_decision": side,
                         "protected_group": ("gender", "religion", "military_status")[i % 3],
                         "protected_attr": f"a{i % 3}",
                         "condition": ("explicit", "implicit")[i % 2],
                         "prompt": f"prompt {side} {i}", "tipped": i < tipped})
    return pd.DataFrame(rows)


# ---- prompt set ---------------------------------------------------------------------------

def test_train_set_is_exactly_half_each_verdict_with_the_asked_tipped_share():
    rows = _rows(n_hire=60, n_reject=400, tipped_hire=30, tipped_reject=100)
    train, plan = G.balanced_train(rows, 40, tipped_share=0.5, seed=1)
    assert train["reference_decision"].value_counts().to_dict() == {"hire": 20, "reject": 20}
    for side in ("hire", "reject"):
        assert train[train["reference_decision"] == side]["tipped"].sum() == 10
        assert plan[side]["drawn_tipped"] == 10 and plan[side]["drawn_untipped"] == 10
    assert not train.duplicated(G.ROW_KEY).any()


def test_a_side_short_of_tipped_rows_fills_with_untipped_ones():
    rows = _rows(n_hire=60, n_reject=60, tipped_hire=3, tipped_reject=30)
    train, plan = G.balanced_train(rows, 40, tipped_share=0.5)
    assert plan["hire"] == {"available": 60, "available_tipped": 3, "drawn_tipped": 3,
                            "drawn_untipped": 17}
    assert (train["reference_decision"] == "hire").sum() == 20


def test_a_side_too_small_for_its_half_refuses():
    with pytest.raises(ValueError, match="hire-reference rows"):
        G.balanced_train(_rows(5, 100, 2, 10), 40, tipped_share=0.5)
    with pytest.raises(ValueError, match="in half"):
        G.balanced_train(_rows(50, 50, 0, 0), 41, tipped_share=0.5)


def test_the_draw_is_seeded():
    rows = _rows(80, 80, 20, 20)
    a, _ = G.balanced_train(rows, 40, 0.5, seed=7)
    b, _ = G.balanced_train(rows, 40, 0.5, seed=7)
    c, _ = G.balanced_train(rows, 40, 0.5, seed=8)
    assert a.equals(b) and not a.equals(c)


def test_held_out_candidates_never_reach_training():
    rows = _rows(60, 60, 10, 10)
    held = {"hire-c0", "reject-c3"}
    train_pool, held_pool = G.split_by_candidate(rows, held)
    assert set(held_pool["candidate_id"]) == held
    assert not set(train_pool["candidate_id"]) & held
    validation = G.balanced_validation(held_pool, 4)
    assert validation["reference_decision"].value_counts().to_dict() == {"hire": 2, "reject": 2}


def test_tipped_flags_are_the_union_of_both_selections():
    rows = _rows(3, 3, 0, 0)
    k8 = pd.DataFrame([{**rows.iloc[0][G.ROW_KEY].to_dict(), "tipped": True},
                       {**rows.iloc[1][G.ROW_KEY].to_dict(), "tipped": False}])
    k32 = pd.DataFrame([{**rows.iloc[4][G.ROW_KEY].to_dict(), "tipped": True}])
    marked = G.mark_tipped(rows, G.tipped_keys(k8, k32))
    assert marked["tipped"].tolist() == [True, False, False, False, True, False]


def test_prompts_attach_by_row_position_and_refuse_a_mismatch():
    rows = _rows(2, 1, 0, 0).drop(columns="prompt")
    prompts = pd.DataFrame({"kind": ["attr_free", "attributed", "attributed", "attributed"],
                            "key": ["hire-p0", "2", "0", "1"],
                            "prompt": ["free", "row two", "row zero", "row one"]})
    assert G.attach_prompts(rows, prompts)["prompt"].tolist() == ["row zero", "row one",
                                                                  "row two"]
    with pytest.raises(ValueError, match="different runs"):
        G.attach_prompts(rows.iloc[:2], prompts)


def _with_pull(rows: pd.DataFrame, rate_tipped: dict, rate_untipped: dict) -> pd.DataFrame:
    """Every row's student hire rate set by verdict and tipped flag."""
    rate = [(rate_tipped if t else rate_untipped)[r]
            for r, t in zip(rows["reference_decision"], rows["tipped"])]
    counts = {p: (round(h * 8), 8) for p, h in zip(rows["prompt"], rate)}
    return G.attach_sft_rates(rows, counts)


def test_student_samples_give_each_row_its_rate_and_pull():
    rows = _rows(2, 2, 0, 0)
    samples = pd.DataFrame({
        "prompt": [rows["prompt"][0]] * 4 + [rows["prompt"][1]] * 2 + [rows["prompt"][2]] * 2
        + [rows["prompt"][3]] * 2,
        "raw_output": [_answer("hire")] * 3 + [_answer("reject"), _answer("hire"),
                                               "no answer", _answer("hire"), _answer("reject"),
                                               _answer("reject"), _answer("reject")]})
    k8 = G.sample_hire_counts(samples)
    k32 = {rows["prompt"][1]: (16, 32)}
    out = G.attach_sft_rates(rows, k32, k8)
    assert out["sft_hire_rate"].tolist() == [0.75, 0.5, 0.5, 0.0]
    assert out["sft_samples"].tolist() == [4, 32, 2, 2]
    # hire rows err by rejecting, reject rows by hiring; pull = sqrt(p(1-p))
    assert out["pull"].tolist() == pytest.approx([0.75 ** 0.5 * 0.5, 0.5, 0.5, 0.0])
    with pytest.raises(KeyError, match="no stored student samples"):
        G.attach_sft_rates(rows, k32)


def test_signal_balance_gives_both_halves_the_same_pull():
    # hire side: few errors per row (pull 0.33 tipped); reject side: many (pull 0.48 tipped)
    rows = _with_pull(_rows(100, 400, 60, 200), {"hire": 7 / 8, "reject": 3 / 8},
                      {"hire": 1.0, "reject": 0.0})
    train, plan = G.signal_balanced_train(rows, 80, seed=3)
    assert train["reference_decision"].value_counts().to_dict() == {"hire": 40, "reject": 40}
    assert plan["hire"]["drawn_tipped"] == 40  # the weaker side takes every tipped row it can
    assert plan["reject"]["drawn_tipped"] < 40
    pulls = train.groupby("reference_decision")["pull"].sum()
    assert abs(pulls["hire"] - pulls["reject"]) <= 0.5  # one reject row's pull
    assert plan["hire"]["pull"] == pytest.approx(pulls["hire"], abs=0.01)
    assert not train.duplicated(G.ROW_KEY).any()


def test_signal_balance_refuses_a_side_too_small_and_is_seeded():
    rows = _with_pull(_rows(30, 400, 10, 100), {"hire": 0.5, "reject": 0.5},
                      {"hire": 1.0, "reject": 0.0})
    with pytest.raises(ValueError, match="hire-reference rows"):
        G.signal_balanced_train(rows, 80)
    rows = _with_pull(_rows(100, 100, 50, 50), {"hire": 0.75, "reject": 0.5},
                      {"hire": 1.0, "reject": 0.0})
    a, _ = G.signal_balanced_train(rows, 40, seed=5)
    b, _ = G.signal_balanced_train(rows, 40, seed=5)
    assert a.equals(b)


# ---- rewards ------------------------------------------------------------------------------

def _answer(decision: str, feedback: str = "Strong match for the stack and seniority asked") -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def test_decision_reward_is_the_attribute_free_reference_in_both_directions():
    completions = [_answer("hire"), _answer("reject"), _answer("reject"), _answer("hire"),
                   "I cannot decide this.", _answer("Hire")]
    reference = ["hire", "hire", "reject", "reject", "hire", "HIRE"]
    rewards = grpo.decision_reward(completions, reference, lang=["en"] * 6)
    assert rewards == [1.0, 0.0, 1.0, 0.0, 0.0, 1.0]


def test_decision_reward_reports_the_sampled_hire_rate_per_reference_side():
    seen = {}
    grpo.decision_reward([_answer("hire"), _answer("reject"), _answer("hire")],
                         ["hire", "hire", "reject"], log_metric=seen.__setitem__)
    assert seen == {"sampled_hire_rate_ref_hire": 0.5, "sampled_hire_rate_ref_reject": 1.0}


@pytest.mark.parametrize("completion, ok", [
    (_answer("reject"), True),
    ("  " + _answer("hire") + "\n", True),
    ("```json\n" + _answer("hire") + "\n```", False),
    ("Decision: " + _answer("hire"), False),
    (json.dumps({"decision": "hire", "feedback": "Good fit.", "score": 3}), False),
    (json.dumps({"decision": "hire"}), False),
    (json.dumps({"decision": "hire", "feedback": ["a", "b"]}), False),
    (_answer("hire", "Too short"), False),
    (_answer("hire", "one two three four"), True),
    (_answer("hire", " ".join(["word"] * 45)), True),
    (_answer("hire", " ".join(["word"] * 46)), False),
    ('{"decision": "hire", "feedback": "unterminated', False),
])
def test_format_reward_wants_exactly_the_json_answer(completion, ok):
    assert grpo.format_reward([completion]) == [grpo.FORMAT_REWARD if ok else 0.0]


def test_rewards_ignore_the_extra_columns_trl_passes():
    kwargs = {"prompts": ["p"], "completion_ids": [[1]], "pair_id": ["x"],
              "protected_group": ["gender"], "trainer_state": None, "log_extra": None}
    assert grpo.decision_reward([_answer("hire")], ["hire"], **kwargs) == [1.0]
    assert grpo.format_reward([_answer("hire")], reference_decision=["hire"], **kwargs) == [0.2]


# ---- weights to vLLM ----------------------------------------------------------------------

@pytest.mark.parametrize("name, multimodal, expected", [
    ("model.layers.3.mlp.up_proj.weight", True, "model.language_model.layers.3.mlp.up_proj.weight"),
    ("model.embed_tokens.weight", True, "model.language_model.embed_tokens.weight"),
    ("model.language_model.norm.weight", True, "model.language_model.norm.weight"),
    ("model.visual.blocks.0.attn.qkv.weight", True, "model.visual.blocks.0.attn.qkv.weight"),
    ("lm_head.weight", True, "lm_head.weight"),
    ("model.layers.3.mlp.up_proj.weight", False, "model.layers.3.mlp.up_proj.weight"),
])
def test_text_model_names_become_checkpoint_names(name, multimodal, expected):
    assert grpo.vllm_param_name(name, multimodal) == expected


def _tiny_peft():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from torch import nn

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.up_proj = nn.Linear(8, 16, bias=True)
            self.norm = nn.LayerNorm(8)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.lm_head = nn.Linear(8, 4, bias=False)

    torch.manual_seed(0)
    model = Tiny().to(torch.bfloat16)
    config = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "up_proj"],
                             lora_dropout=0.0)
    model = peft.get_peft_model(model, config)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 0.5)
    return model


def test_merged_weights_match_peft_merge_and_leave_the_base_untouched():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    pushed = dict(grpo.merged_named_params(model, multimodal=True))

    reference = copy.deepcopy(model).float().merge_and_unload().state_dict()
    expected = {grpo.vllm_param_name(k, True): v for k, v in reference.items()}
    assert set(pushed) == set(expected)
    for name, value in pushed.items():
        assert value.dtype == torch.bfloat16
        assert torch.allclose(value.float(), expected[name].to(torch.bfloat16).float(),
                              atol=1e-2), name
    assert not any("lora_" in n or "base_layer" in n for n in pushed)
    assert all(n.startswith(("model.language_model.", "lm_head.")) for n in pushed)
    lora = "model.language_model.layers.0.q_proj.weight"
    base = before["base_model.model.model.layers.0.q_proj.base_layer.weight"]
    assert not torch.equal(pushed[lora], base)  # the adapter is in what vLLM gets
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} was written"


def test_a_zero_adapter_pushes_the_base_weights_exactly():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
    pushed = dict(grpo.merged_named_params(model))
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        clean = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        assert torch.equal(pushed[clean], param), clean


# ---- decision monitor ---------------------------------------------------------------------

def test_projected_hire_rate_carries_relative_changes_onto_sfts_benchmark_rates():
    sft = {"hire_rate_ref_hire": 1.0, "hire_rate_ref_reject": 0.21875}
    same = grpo.projected_hire_rate(sft, sft, n_side=32)
    assert same["hire_rate_projected"] == pytest.approx(0.333 * 0.658 + 0.667 * 0.091)
    # smoke job 5065891 at step 20: fewer false hires, three fewer true hires of 32
    later = {"hire_rate_ref_hire": 0.90625, "hire_rate_ref_reject": 0.0625}
    out = grpo.projected_hire_rate(later, sft, n_side=32)
    assert out["projected_hire_rate_ref_hire"] == pytest.approx(0.658 * 29.5 / 32.5)
    assert out["projected_hire_rate_ref_reject"] == pytest.approx(0.091 * 2.5 / 7.5)
    assert 0.20 < out["hire_rate_projected"] < 0.23
    # r1-300's own change on the benchmark lands on its own hire rate
    r1 = {"hire_rate_ref_hire": 0.476 / 0.658, "hire_rate_ref_reject": 0.048 / 0.091}
    base = {"hire_rate_ref_hire": 1.0, "hire_rate_ref_reject": 1.0}
    assert grpo.projected_hire_rate(r1, base, n_side=10_000)["hire_rate_projected"] == \
        pytest.approx(0.191, abs=0.002)


def test_the_monitor_projects_onto_its_own_sft_parents_benchmark_rates():
    pytest.importorskip("transformers")
    rows = pd.DataFrame({"reference_decision": ["hire", "reject"]})
    default = grpo._decision_monitor_class()(rows, None, ".", 25, (0.2, 0.4), None)
    assert default.sft_benchmark == grpo.SFT_BENCHMARK
    # SFT-v3 (reports/reference_metrics.csv): 57.73% where GPT-4o hires, 5.266% where it rejects
    v3 = {"hire_rate_ref_hire": 0.5773, "hire_rate_ref_reject": 0.05266}
    monitor = grpo._decision_monitor_class()(rows, None, ".", 25, (0.15, 0.35), None,
                                             sft_benchmark=v3)
    assert monitor.sft_benchmark["reference_hire_share"] == 0.333
    same = {"hire_rate_ref_hire": 1.0, "hire_rate_ref_reject": 0.2}
    out = grpo.projected_hire_rate(same, same, 128, monitor.sft_benchmark)
    assert out["hire_rate_projected"] == pytest.approx(0.333 * 0.5773 + 0.667 * 0.05266)


def test_merged_into_base_runs_the_pushed_weights_and_restores_the_base_bit_for_bit():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    pushed = dict(grpo.merged_named_params(model))
    with grpo.merged_into_base(model), model.disable_adapter():
        for name, module in grpo.lora_layers(model):
            clean = name.removeprefix("base_model.model.") + ".weight"
            assert torch.equal(module.base_layer.weight, pushed[clean])
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name]), f"{name} not restored"


def test_merge_retention_sees_a_delta_too_small_for_bf16():
    torch = pytest.importorskip("torch")
    model = _tiny_peft()
    assert grpo.merge_retention(model)["kept_fraction"] == pytest.approx(1.0, abs=0.05)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.mul_(1e-5)  # far below half a bf16 unit of the base weights
    tiny = grpo.merge_retention(model)
    assert tiny["kept_fraction"] < 0.2 and tiny["unchanged"] > 0.9
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.zero_()
    assert grpo.merge_retention(model) == {"kept_fraction": 1.0, "unchanged": 1.0}
