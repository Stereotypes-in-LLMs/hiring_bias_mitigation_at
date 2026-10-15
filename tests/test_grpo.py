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
                         "protected_attr": f"a{i % 3}", "condition": ("explicit", "implicit")[i % 2],
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

def test_projected_hire_rate_moves_sfts_benchmark_rate_by_the_reweighted_change():
    sft = {"hire_rate_ref_hire": 0.80, "hire_rate_ref_reject": 0.10}
    later = {"hire_rate_ref_hire": 0.50, "hire_rate_ref_reject": 0.05}
    out = grpo.projected_hire_rate(later, sft, anchor=0.28, ref_share=0.33)
    assert out["hire_rate_weighted_sft"] == pytest.approx(0.33 * 0.80 + 0.67 * 0.10)
    assert out["hire_rate_projected"] == pytest.approx(0.28 + 0.33 * -0.30 + 0.67 * -0.05)
    assert grpo.projected_hire_rate(sft, sft, 0.28, 0.33)["hire_rate_projected"] == \
        pytest.approx(0.28)
