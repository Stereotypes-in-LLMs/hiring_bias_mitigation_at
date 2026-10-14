"""Merging a DPO adapter that was fitted on top of an SFT adapter.

DPO continuing from SFT merges the SFT adapter into the base in memory and fits a fresh LoRA on
that. PEFT then records the plain base as the new adapter's `base_model_name_or_path`, so
rebuilding the model from the adapter alone drops the SFT weights. These tests use a two-layer
model on CPU.
"""

import json

import pytest

from hiring_bias_mitigation.mitigation import merge as MG


def _adapter_dir(path, base="Qwen/Qwen3.5-9B", sft_checkpoint=None):
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": base}))
    if sft_checkpoint is not None:
        (path / "run_config.json").write_text(json.dumps({"sft_checkpoint": sft_checkpoint}))
    return path


def test_a_plain_adapter_is_its_own_chain(tmp_path):
    sft = _adapter_dir(tmp_path / "sft")
    assert MG.adapter_chain(sft) == [sft]


def test_a_dpo_run_and_its_checkpoints_put_the_sft_adapter_first(tmp_path):
    sft = _adapter_dir(tmp_path / "sft")
    dpo = _adapter_dir(tmp_path / "dpo", sft_checkpoint=str(sft))
    checkpoint = _adapter_dir(dpo / "checkpoint-250")
    assert MG.adapter_chain(dpo) == [sft, dpo]
    assert MG.adapter_chain(checkpoint) == [sft, checkpoint]


def test_stacked_merge_reproduces_the_model_dpo_trained_and_the_adapter_alone_does_not(
        tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from transformers import AutoModelForCausalLM, Qwen2Config

    from hiring_bias_mitigation.mitigation import training_common as C

    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         max_position_embeddings=64)
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path / "base")

    def fit_lora(model, path):
        lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"])
        model = peft.get_peft_model(model, lora)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, std=0.5)  # trained: B is no longer zero
        model.save_pretrained(path)
        return model.eval()

    fit_lora(AutoModelForCausalLM.from_pretrained(tmp_path / "base"), tmp_path / "sft")
    # What load_model_and_tokenizer does for a DPO config with an sft_checkpoint.
    source, adapter = C._resolve_adapter(str(tmp_path / "sft"))
    start = peft.PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(source),
                                           adapter).merge_and_unload()
    dpo = fit_lora(start, tmp_path / "dpo")
    (tmp_path / "dpo" / "run_config.json").write_text(
        json.dumps({"sft_checkpoint": str(tmp_path / "sft")}))

    tokens = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        trained = dpo(tokens).logits
        chain = MG.adapter_chain(tmp_path / "dpo")
        stacked = MG.merge_chain(AutoModelForCausalLM.from_pretrained(tmp_path / "base"),
                                 chain)(tokens).logits
        alone = MG.merge_chain(AutoModelForCausalLM.from_pretrained(tmp_path / "base"),
                               [tmp_path / "dpo"])(tokens).logits
    assert chain == [tmp_path / "sft", tmp_path / "dpo"]
    assert torch.allclose(trained, stacked, atol=1e-5)
    assert not torch.allclose(trained, alone, atol=1e-2)
