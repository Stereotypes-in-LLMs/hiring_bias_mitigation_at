"""Decision-level evaluation of DPO checkpoints: what the model decides, not its loss.

No GPU, no network: generation and W&B are replaced by canned outputs and a recorder.
"""

import json
import sys
from types import SimpleNamespace

import pandas as pd

from hiring_bias_mitigation.mitigation import decision_eval as DE


def _out(decision: str) -> str:
    return json.dumps({"decision": decision, "feedback": "Matches the stack."})


def _rows() -> pd.DataFrame:
    return pd.DataFrame({
        "prompt": ["p0", "p1", "p2", "p3"],
        "reference_decision": ["hire", "hire", "reject", "reject"],
        "protected_group": ["gender", "religion", "gender", "religion"],
        "protected_attr": ["Woman", "Jew", "Man", "Muslim"],
        "condition": ["explicit"] * 4,
    })


def test_eval_rows_are_balanced_by_reference_and_reproducible():
    frame = pd.DataFrame({"prompt": [f"p{i}" for i in range(10)],
                          "reference_decision": ["hire"] * 3 + ["reject"] * 7})
    rows = DE.eval_rows(frame, 4, seed=1)
    assert rows["reference_decision"].value_counts().to_dict() == {"hire": 2, "reject": 2}
    assert rows["prompt"].tolist() == DE.eval_rows(frame, 4, seed=1)["prompt"].tolist()
    assert len(DE.eval_rows(frame, None)) == 10


def test_metrics_count_agreement_operating_point_and_groups():
    outputs = [_out("hire"), _out("reject"), _out("hire"), "no json here"]
    m = DE.decision_metrics(_rows(), outputs)
    assert m["n"] == 4
    assert m["agree_reference"] == 0.25
    assert m["hire_rate"] == 0.5
    assert m["hire_rate_ref_hire"] == 0.5
    assert m["hire_rate_ref_reject"] == 0.5
    assert m["invalid_rate"] == 0.25
    assert m["agree_reference/gender"] == 0.5
    assert m["agree_reference/religion"] == 0.0


def test_callback_records_decisions_on_rank_zero_only(tmp_path, monkeypatch):
    """Under DDP every rank runs the callback; one must generate and log, the rest wait."""
    logged = []
    fake_wandb = SimpleNamespace(run=object(), log=lambda data, **kw: logged.append(data),
                                 Table=lambda columns, data: ("table", columns, data))
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(DE, "generate", lambda model, tokenizer, prompts, **kw:
                        [_out("hire"), _out("hire"), _out("reject"), _out("reject")])
    callback = DE.DecisionEvalCallback(_rows(), tokenizer=None, output_dir=tmp_path)

    callback.on_evaluate(None, SimpleNamespace(is_world_process_zero=False, global_step=50),
                         None, model=None)
    assert logged == [] and not (tmp_path / "decision_eval.jsonl").exists()

    callback.on_evaluate(None, SimpleNamespace(is_world_process_zero=True, global_step=50),
                         None, model=None)
    record = json.loads((tmp_path / "decision_eval.jsonl").read_text().splitlines()[0])
    assert record["step"] == 50 and record["metrics"]["agree_reference"] == 1.0
    assert logged[0]["decision/agree_reference"] == 1.0
    assert logged[0]["train/global_step"] == 50
    assert logged[0]["decision/samples"][0] == "table"


def test_checkpoint_steps_start_at_the_sft_model_and_skip_unfinished_saves(tmp_path):
    for step, finished in ((50, True), (25, True), (75, False)):
        (tmp_path / f"checkpoint-{step}").mkdir()
        if finished:
            (tmp_path / f"checkpoint-{step}" / "trainer_state.json").write_text("{}")
    assert DE.checkpoint_steps(tmp_path) == [0, 25, 50]


def test_dpo_attaches_the_decision_callback_only_when_configured(tmp_path):
    from datasets import Dataset

    from hiring_bias_mitigation.mitigation import dpo

    splits = {"train": Dataset.from_pandas(_rows()), "validation": Dataset.from_pandas(_rows())}
    assert dpo.decision_callbacks({}, splits, None, tmp_path) == []
    assert dpo.decision_callbacks({"decision_eval_rows": 2}, {"train": splits["train"]},
                                  None, tmp_path) == []
    [callback] = dpo.decision_callbacks({"decision_eval_rows": 2}, splits, None, tmp_path)
    assert isinstance(callback, DE.DecisionEvalCallback)
    assert len(callback.rows) == 2
