"""Rejection-sampling SFT control: the student's right samples as targets, and a second SFT
stage that starts from the SFT adapter."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from hiring_bias_mitigation.generation import grpo_prompts as G
from hiring_bias_mitigation.generation import rft_data as R


def _answer(decision: str, feedback: str = "Strong match for the stack and seniority asked") -> str:
    return json.dumps({"decision": decision, "feedback": feedback})


def _rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"prompt": "p0", "reference_decision": "hire", "protected_attr": "War veteran",
         "protected_group": "military_status"},
        {"prompt": "p1", "reference_decision": "reject", "protected_attr": "Muslim",
         "protected_group": "religion"},
        {"prompt": "p2", "reference_decision": "reject", "protected_attr": "Female",
         "protected_group": "gender"},
    ])


def test_the_target_is_the_first_right_sample_that_passes_the_filters():
    samples = pd.DataFrame([
        # p0: sample 0 wrong, sample 1 names the attribute, sample 3 right but later than 2
        ("p0", "0", _answer("reject")),
        ("p0", "1", _answer("hire", "A war veteran with strong Java and Spring experience")),
        ("p0", "3", _answer("hire", "Later right answer with Java and Spring experience")),
        ("p0", "2", _answer("hire", "Solid Java and Spring experience fits the role well")),
        # p1: only right answer names the group; then a meta-commentary one
        ("p1", "0", _answer("reject", "Religion aside, the CV lacks the required SQL skills")),
        ("p1", "1", _answer("reject", "Regardless of background, the CV lacks SQL and ETL")),
        # p2: never right
        ("p2", "0", _answer("hire")),
    ], columns=["prompt", "sample_index", "raw_output"])
    found, report = R.first_right_samples(_rows(), samples, "k8")
    assert found["_row"].tolist() == [0]
    assert found.iloc[0]["sample_index"] == 2
    assert found.iloc[0]["feedback"] == "Solid Java and Spring experience fits the role well"
    assert report["rows_with_a_target"] == 1 and report["rows_with_samples"] == 3
    assert report["group_leak"] == 1
    assert report["dropped_by_reason"]["invariant_attribute_leak"] == 1
    assert report["dropped_by_reason"]["invariant_meta_commentary"] == 1


def test_targets_prefer_the_larger_sample_and_carry_the_row_error():
    rows = _rows().assign(sft_hire_rate=[0.75, 0.25, 1.0], sft_samples=[32, 8, 8])
    k32 = pd.DataFrame({"_row": [0], "sample_index": [5], "decision": ["hire"],
                        "feedback": ["from the top-up"], "sample_source": ["k32"]})
    k8 = pd.DataFrame({"_row": [0, 1], "sample_index": [0, 1], "decision": ["hire", "reject"],
                       "feedback": ["from the pool", "pool reject"], "sample_source": ["k8"] * 2})
    out = R.attach_targets(rows, k32, k8)
    assert out["prompt"].tolist() == ["p0", "p1"]  # p2 has no target
    assert out["sample_source"].tolist() == ["k32", "k8"]
    assert json.loads(out["completion"][0]) == {"decision": "hire", "feedback": "from the top-up"}
    assert out["error"].tolist() == pytest.approx([0.25, 0.25])


def test_rft_halves_are_balanced_by_error_mass():
    rows = pd.DataFrame(
        [{"prompt": f"h{i}", "reference_decision": "hire", "tipped": i < 30,
          "error": 0.1 if i < 30 else 0.0, "pair_id": f"h{i}", "protected_attr": "a",
          "condition": "explicit"} for i in range(60)]
        + [{"prompt": f"r{i}", "reference_decision": "reject", "tipped": i < 50,
            "error": 0.3 if i < 50 else 0.0, "pair_id": f"r{i}", "protected_attr": "a",
            "condition": "explicit"} for i in range(200)])
    rows["sft_hire_rate"] = rows["error"].where(rows["reference_decision"] == "reject",
                                                1 - rows["error"])
    train, plan = G.signal_balanced_train(rows, 40, seed=1, column="error")
    mass = train.groupby("reference_decision")["error"].sum()
    assert plan["hire"]["drawn_tipped"] == 20
    assert mass["hire"] == pytest.approx(2.0) and abs(mass["reject"] - 2.0) <= 0.3


# ---- a second SFT stage from the SFT adapter ----------------------------------------------

def test_sft_main_starts_from_the_sft_checkpoint(monkeypatch, tmp_path):
    from hiring_bias_mitigation.mitigation import sft as S
    from hiring_bias_mitigation.mitigation import training_common as C

    adapter = tmp_path / "sft-adapter"
    config = tmp_path / "rft.yaml"
    config.write_text("stage: sft\nbase_model: Qwen/Qwen3.5-9B\n"
                      "sft_checkpoint: Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en\n"
                      "output_dir: outputs/sft/x\n")
    seen = {}
    monkeypatch.setattr(C, "resolve_sft_checkpoint", lambda spec: str(adapter))
    monkeypatch.setattr(S, "train",
                        lambda cfg: seen.update(cfg) or C.TrainArtifacts(tmp_path, 0, 0))
    monkeypatch.setattr("sys.argv", ["sft", "--config", str(config)])
    S.main()
    assert seen["base_model"] == str(adapter)
    assert seen["sft_checkpoint"] == "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en"


def test_a_second_sft_stage_merges_with_its_sft_parent(monkeypatch, tmp_path):
    from hiring_bias_mitigation.mitigation import merge as MG
    from hiring_bias_mitigation.mitigation import training_common as C

    sft, rft = tmp_path / "sft", tmp_path / "rft"
    for a in (sft, rft):
        a.mkdir()
        (a / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}))
    (rft / "checkpoint-25").mkdir()
    cfg = {"base_model": str(sft), "stage": "sft", "data": {}}
    C.write_run_manifest(cfg, rft, {"mitigation_family": "sft", "sft_checkpoint": str(sft)})
    monkeypatch.setattr(C, "resolve_sft_checkpoint", lambda spec: spec)
    assert MG.adapter_chain(rft) == [sft, rft]
    assert MG.adapter_chain(rft / "checkpoint-25") == [sft, rft / "checkpoint-25"]
    # a first-stage SFT run records no parent and stays a chain of one
    C.write_run_manifest(cfg, sft, {"mitigation_family": "sft", "sft_checkpoint": None})
    assert MG.adapter_chain(sft) == [sft]


def test_sft_leaves_an_unset_loss_type_to_trl():
    """TRL 1.15 rejects loss_type=None; job 5066765 died on it before its first step."""
    from hiring_bias_mitigation.mitigation.sft import sft_config_candidates

    assert "loss_type" not in sft_config_candidates({"max_seq_len": 2048})
    assert sft_config_candidates({}, weight=3.0)["loss_type"] == "nll"
    assert sft_config_candidates({"loss_type": "dft"})["loss_type"] == "dft"
    assert sft_config_candidates({})["completion_only_loss"] is True
