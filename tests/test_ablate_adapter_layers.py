"""Removing a trained adapter's update in some layers (scripts/ablate_adapter_layers.py)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

_spec = importlib.util.spec_from_file_location(
    "ablate_adapter_layers",
    Path(__file__).resolve().parents[1] / "scripts" / "ablate_adapter_layers.py")
AB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AB)

PREFIX = "base_model.model.model.language_model.layers"


def _adapter(tmp_path: Path) -> Path:
    """Three decoder layers, two modules each, plus the files the merge chain reads."""
    rng = np.random.default_rng(0)
    tensors = {}
    for layer in (14, 15, 21):
        for module in ("mlp.down_proj", "self_attn.q_proj"):
            key = f"{PREFIX}.{layer}.{module}"
            tensors[f"{key}.lora_A.weight"] = rng.normal(size=(2, 6)).astype(np.float32)
            tensors[f"{key}.lora_B.weight"] = rng.normal(size=(5, 2)).astype(np.float32)
    adapter = tmp_path / "decision_distill_seed42"
    adapter.mkdir()
    save_file(tensors, str(adapter / "adapter_model.safetensors"))
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    (adapter / "run_config.json").write_text(json.dumps({"sft_checkpoint": "outputs/sft/x"}))
    return adapter


def test_zeroes_only_lora_B_in_the_window(tmp_path):
    adapter = _adapter(tmp_path)
    out = tmp_path / "decision_distill_seed42_xw15_20"
    record = AB.ablate(adapter, out, 15, 20)
    before = load_file(str(adapter / "adapter_model.safetensors"))
    after = load_file(str(out / "adapter_model.safetensors"))
    assert set(after) == set(before)
    for name, value in after.items():
        if ".layers.15." in name and ".lora_B." in name:
            assert not value.any()
        else:
            np.testing.assert_array_equal(value, before[name])
    assert record["layers_found"] == [15]
    assert record["modules_total"] == 6
    assert record["modules_zeroed"] == 2
    assert record["lora_B_params_zeroed"] == 2 * 5 * 2
    # The merge chain reads the parent from run_config.json, so the copy stacks the same way.
    for name in ("adapter_config.json", "run_config.json"):
        assert (out / name).read_text() == (adapter / name).read_text()
    assert json.loads((out / "ablation.json").read_text())["layers_zeroed"] == [15, 20]


def test_refuses_an_empty_window_and_an_existing_output(tmp_path):
    adapter = _adapter(tmp_path)
    with pytest.raises(ValueError, match="no LoRA module"):
        AB.ablate(adapter, tmp_path / "none", 16, 20)
    out = tmp_path / "exists"
    out.mkdir()
    with pytest.raises(FileExistsError):
        AB.ablate(adapter, out, 15, 20)
