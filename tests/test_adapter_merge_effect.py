"""Guard the comparison's chain, resumability and complete-pair pilot selection."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "merge_effect", Path(__file__).parents[1] / "scripts/check_adapter_merge_effect.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_chain_rejects_double_sft_or_wrong_served_checkpoint(tmp_path):
    sft, adapter = tmp_path / "sft", tmp_path / "last"
    marker = {"adapters": [str(sft), str(adapter)]}
    M.validate_chain([sft, adapter], sft, adapter, marker)
    with pytest.raises(ValueError, match="exactly common SFT"):
        M.validate_chain([sft, sft, adapter], sft, adapter, marker)
    with pytest.raises(ValueError, match="different adapter chain"):
        M.validate_chain([sft, adapter], sft, adapter, {"adapters": [str(adapter)]})


def test_resume_refuses_changed_scoring_inputs(tmp_path):
    path = tmp_path / "identity.json"
    M.freeze(path, {"batch": 8, "adapter_sha": "a"})
    M.freeze(path, {"adapter_sha": "a", "batch": 8})
    with pytest.raises(ValueError, match="stale resume"):
        M.freeze(path, {"batch": 16, "adapter_sha": "a"})
    assert json.loads(path.read_text())["batch"] == 8


def test_pilot_keeps_all_variants_and_covers_lengths():
    rows = [{"pair_id": i, "text": "x" * (i + 1), "variant": v}
            for i in range(20) for v in range(5)]
    frame = pd.DataFrame(rows)
    selected = M.select_pilot(frame, 4)
    assert selected.pair_id.nunique() == 4
    assert selected.groupby("pair_id").size().eq(5).all()
    assert {0, 19}.issubset(set(selected.pair_id))
    pd.testing.assert_frame_equal(selected, M.select_pilot(frame, 4))


def test_score_comparison_requires_identical_prompt_order():
    a = pd.DataFrame({"prompt_uid": ["a", "b"], "margin_fp32": [-1., 1.]})
    b = pd.DataFrame({"prompt_uid": ["a", "b"], "margin_fp32": [1., 1.]})
    assert M.summary(a, b, "margin_fp32")["decision_agreement_pct"] == 50
    with pytest.raises(ValueError, match="rows differ"):
        M.summary(a, b.iloc[::-1], "margin_fp32")


def test_weight_parity_checks_every_parameter_and_detects_rounding(tmp_path):
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("safetensors.torch")
    model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    sf.save_file(model.state_dict(), tmp_path / "model.safetensors")
    assert M.verify_saved_weights(model, tmp_path)["all_unique_parameters_bit_identical"]
    with torch.no_grad():
        model.weight[0, 0] += 1
    with pytest.raises(ValueError, match="differs from served"):
        M.verify_saved_weights(model, tmp_path)


def test_completed_artifacts_cannot_hide_missing_rows_or_changed_values(tmp_path):
    pytest.importorskip("pyarrow")
    (tmp_path / "unmerged").mkdir()
    path = tmp_path / "unmerged/part-00000.parquet"
    frame = pd.DataFrame({"prompt_uid": ["a", "b"], **{key: [1., 2.] for key in M.READOUTS}})
    frame.to_parquet(path, index=False)
    first = M.score_artifacts(tmp_path, ["unmerged"], ["a", "b"])
    frame.loc[0, "margin_fp32"] = 3
    frame.to_parquet(path, index=False)
    assert M.score_artifacts(tmp_path, ["unmerged"], ["a", "b"]) != first
    frame.iloc[:1].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Incomplete completed"):
        M.score_artifacts(tmp_path, ["unmerged"], ["a", "b"])


def test_qwen_text_namespace_is_mapped_without_relaxing_weight_parity(tmp_path):
    torch = pytest.importorskip("torch")
    sf = pytest.importorskip("safetensors.torch")
    model = torch.nn.Module()
    model.model = torch.nn.Linear(2, 2).to(torch.bfloat16)
    state = {k.replace("model.", "model.language_model.", 1): v for k, v in model.state_dict().items()}
    sf.save_file(state, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="missing parameter"):
        M.verify_saved_weights(model, tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Qwen3_5ForConditionalGeneration"]}))
    assert M.verify_saved_weights(model, tmp_path)["qwen_text_prefix_mappings"] == 2
    with torch.no_grad():
        model.model.weight[0, 0] += 1
    with pytest.raises(ValueError, match="differs from served"):
        M.verify_saved_weights(model, tmp_path)
