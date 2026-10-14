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


def test_training_arguments_take_save_and_ddp_overrides(tmp_path):
    """Segmented runs on a 30-minute partition save more often than they evaluate, which
    load_best_model_at_end forbids; DDP with LoRA needs non-reentrant checkpointing."""
    from hiring_bias_mitigation.mitigation import training_common as C

    base = {"eval_steps": 50, "dtype": "float32", "use_wandb": False}
    args = C.training_arguments(
        {**base, "save_steps": 25, "save_total_limit": 20, "load_best_model_at_end": False,
         "ddp_find_unused_parameters": False,
         "gradient_checkpointing_kwargs": {"use_reentrant": False}},
        tmp_path, has_eval=True, n_train=100)
    assert (args.save_steps, args.eval_steps, args.save_total_limit) == (25, 50, 20)
    assert args.load_best_model_at_end is False
    assert args.ddp_find_unused_parameters is False
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}

    default = C.training_arguments(base, tmp_path, has_eval=True, n_train=100)
    assert default.save_steps == 50 and default.load_best_model_at_end is True


def test_last_checkpoint_is_the_highest_one_the_trainer_finished_writing(tmp_path):
    from hiring_bias_mitigation.mitigation import training_common as C

    assert C.last_checkpoint(tmp_path / "missing") is None
    assert C.last_checkpoint(tmp_path) is None
    for step, finished in ((25, True), (50, True), (75, False), (100, True)):
        d = tmp_path / f"checkpoint-{step}"
        d.mkdir()
        if finished:
            (d / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-100" / "trainer_state.json").unlink()  # cut mid-save
    assert C.last_checkpoint(tmp_path) == str(tmp_path / "checkpoint-50")


def test_dpo_resumes_from_its_last_checkpoint_and_names_its_parent_before_training(
        tmp_path, monkeypatch):
    """A run cut by the time limit never reaches the final manifest, and without it a
    checkpoint-N adapter would be merged without its SFT parent (merge.adapter_chain)."""
    from datasets import Dataset

    from hiring_bias_mitigation.mitigation import dpo
    from hiring_bias_mitigation.mitigation import training_common as C

    monkeypatch.setenv("HBM_ALLOW_LOCAL_OUTPUT", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    out = tmp_path / "run"
    (out / "checkpoint-25").mkdir(parents=True)
    (out / "checkpoint-25" / "trainer_state.json").write_text("{}")
    rows = {"prompt": ["p"], "chosen": ["c"], "rejected": ["r"]}
    monkeypatch.setattr(C, "load_generated_dataset",
                        lambda cfg, kind: {"train": Dataset.from_dict(rows)})

    class Tokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                                **kwargs):
            text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
            return text + ("<assistant>" if add_generation_prompt else "")

        def save_pretrained(self, path):
            pass

    monkeypatch.setattr(C, "load_model_and_tokenizer", lambda cfg: (object(), Tokenizer()))
    seen = {}

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def train(self, resume_from_checkpoint=None):
            seen["resume"] = resume_from_checkpoint
            seen["manifest"] = json.loads((out / "run_config.json").read_text())

        def save_model(self, path):
            pass

    monkeypatch.setattr("trl.DPOTrainer", Trainer)
    monkeypatch.setattr(C, "wandb_resume", lambda cfg, path: seen.setdefault("wandb_dir", path))
    dpo.train({"objective": "dpo", "output_dir": str(out), "use_wandb": False,
               "dtype": "float32", "use_lora": False, "memory_floor_gb": 0, "resume": True,
               "base_model": "/cache/sft-snapshot", "data": {},
               "sft_checkpoint": "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en"})
    assert seen["resume"] == str(out / "checkpoint-25")
    assert seen["manifest"]["sft_checkpoint"] == \
        "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en"
    assert seen["manifest"]["status"] == "training"
    assert json.loads((out / "run_config.json").read_text())["status"] == "done"
    assert seen["wandb_dir"] == out


def test_precomputed_reference_logps_under_torchrun_are_refused_before_the_model_loads(
        monkeypatch):
    """TRL's precompute writes its cache on rank 0 into the dataset's cache dir, which for an
    in-memory dataset is a per-process temp dir the other ranks cannot read (job 5054142 died
    there after 6 minutes)."""
    import pytest

    from hiring_bias_mitigation.mitigation import dpo
    from hiring_bias_mitigation.mitigation import training_common as C

    def never(*args, **kwargs):
        raise AssertionError("must refuse before loading anything")

    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(C, "load_generated_dataset", never)
    monkeypatch.setattr(C, "load_model_and_tokenizer", never)
    with pytest.raises(ValueError, match="precompute_ref_log_probs"):
        dpo.train({"objective": "dpo", "precompute_ref_log_probs": True, "output_dir": "x"})
