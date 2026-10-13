"""Continuing DPO from the *published* SFT adapter, with the v2 loss options."""

import json


def test_hub_adapter_id_is_fetched_and_resolved_to_a_local_adapter_dir(tmp_path, monkeypatch):
    """`sft_checkpoint: Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en` must work without
    anyone copying the adapter onto the drive first."""
    from hiring_bias_mitigation.mitigation import training_common as C

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B", "r": 32}), encoding="utf-8"
    )
    seen = {}

    def fake_snapshot_download(repo_id, **kwargs):
        seen["repo_id"] = repo_id
        seen["allow_patterns"] = kwargs.get("allow_patterns")
        return str(snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    resolved = C.resolve_sft_checkpoint("Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en")
    assert resolved == str(snapshot)
    assert seen["repo_id"] == "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en"
    assert "adapter_config.json" in seen["allow_patterns"]
    # and the existing local-directory logic then places it on its base model
    assert C._resolve_adapter(resolved) == ("Qwen/Qwen3.5-9B", str(snapshot))


def test_local_sft_checkpoint_still_wins_over_the_hub(tmp_path, monkeypatch):
    from hiring_bias_mitigation.mitigation import training_common as C

    local = tmp_path / "outputs" / "sft" / "run"
    local.mkdir(parents=True)
    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network")))
    assert C.resolve_sft_checkpoint("outputs/sft/run") == str(local)


def test_a_checkpoint_that_is_neither_local_nor_on_the_hub_is_an_error(tmp_path, monkeypatch):
    import pytest

    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))

    def missing(*args, **kwargs):
        raise OSError("404")

    monkeypatch.setattr("huggingface_hub.snapshot_download", missing)
    with pytest.raises(FileNotFoundError, match="sft_checkpoint"):
        C.resolve_sft_checkpoint("outputs/sft/never-trained")


def test_dpo_loss_options_reach_the_trainer_only_when_configured():
    from trl import DPOConfig

    from hiring_bias_mitigation.mitigation import dpo
    from hiring_bias_mitigation.mitigation import training_common as C

    plain = dpo.dpo_config_candidates({"beta": 0.1})
    assert plain["beta"] == 0.1
    assert "loss_type" not in plain and "loss_weights" not in plain

    v2 = dpo.dpo_config_candidates({
        "beta": 0.1, "loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 0.2],
        "ld_alpha": 0.5, "precompute_ref_log_probs": True,
    })
    kept = C.supported_kwargs(DPOConfig, v2, "DPOConfig")
    assert kept["loss_type"] == ["sigmoid", "sft"]
    assert kept["loss_weights"] == [1.0, 0.2]
    assert kept["ld_alpha"] == 0.5
    assert kept["precompute_ref_log_probs"] is True


def test_hub_adapter_fetch_includes_the_tokenizer_the_loader_reads(tmp_path, monkeypatch):
    """dpo.main sets base_model to the fetched adapter folder, and load_model_and_tokenizer
    reads the tokenizer from there. Fetching only the adapter weights crashes on a fresh cache."""
    import fnmatch

    from hiring_bias_mitigation.mitigation import training_common as C

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}), encoding="utf-8")
    seen = {}

    def fake_snapshot_download(repo_id, **kwargs):
        seen["allow_patterns"] = kwargs.get("allow_patterns") or ["*"]
        return str(snapshot)

    monkeypatch.setenv("HBM_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    C.resolve_sft_checkpoint("Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en")
    needed = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json",
              "tokenizer_config.json", "chat_template.jinja"]
    for name in needed:
        assert any(fnmatch.fnmatch(name, pat) for pat in seen["allow_patterns"]), name
