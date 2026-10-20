"""Per-class GRPO signal: the expected-group arithmetic, the step split and the log parser."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "grpo_signal_by_reference.py"
_spec = importlib.util.spec_from_file_location("grpo_signal_by_reference", _path)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def test_a_group_has_no_signal_when_every_completion_agrees():
    out = S.group_stats(np.array([1.0, 0.0, 0.5]), np.array([0.0, 1.0, 0.5]), k=8)
    assert out["zero_variance"] == pytest.approx((1 + 1 + 2 / 256) / 3)
    assert out["all_right"] == pytest.approx((1 + 1 / 256) / 3)
    assert out["all_wrong"] == pytest.approx((1 + 1 / 256) / 3)
    assert out["signal_prompts"] == pytest.approx(1 - 2 / 256)
    assert out["accuracy"] == pytest.approx(0.5)
    assert out["decision_mass"] == pytest.approx(1.0)


def test_mass_off_the_two_decision_tokens_is_reported_not_counted():
    out = S.group_stats(np.array([0.6]), np.array([0.3]), k=2)
    assert out["zero_variance"] == pytest.approx(0.36 + 0.09)
    assert out["decision_mass"] == pytest.approx(0.9)


def test_right_and_wrong_follow_the_reference():
    frame = pd.DataFrame({"reference_decision": ["hire", "reject"],
                          "lp_hire": np.log([0.9, 0.2]), "lp_reject": np.log([0.1, 0.8])})
    p, q = S.right_wrong(frame)
    assert p == pytest.approx([0.9, 0.8]) and q == pytest.approx([0.1, 0.2])


def test_checkpoints_and_shards(tmp_path):
    for step in (50, 25, 187):
        (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / f"checkpoint-{step}" / "adapter_model.safetensors").write_bytes(b"x")
    (tmp_path / "checkpoint-75").mkdir()  # cut mid-save: no weights
    steps = S.checkpoint_steps(tmp_path)
    assert steps == [0, 25, 50, 187]
    shards = [S.shard_steps(steps, i, 3) for i in range(3)]
    assert sorted(s for shard in shards for s in shard) == steps and shards[0] == [0, 187]


def test_the_trainer_log_parses_into_per_step_series():
    text = ("noise {'loss': '0.01', 'kl': '0', 'rewards/decision_reward/mean': '0.75', "
            "'frac_reward_zero_std': '0.5', 'sampled_hire_rate_ref_hire': '0.9', "
            "'sampled_hire_rate_ref_reject': '0.2', 'epoch': '0.01'} more "
            "{'loss': '0.02', 'kl': '0.1', 'rewards/decision_reward/mean': '1', "
            "'frac_reward_zero_std': '1', 'sampled_hire_rate_ref_hire': '1', "
            "'sampled_hire_rate_ref_reject': '0', 'epoch': '0.02'} "
            "{'train_runtime': '100', 'epoch': '1'}")
    log = S.parse_trainer_log(text)
    assert log["step"].tolist() == [1, 2]
    series = S.per_step_series(log)
    assert series["accuracy_reject_logged"].tolist() == pytest.approx([0.8, 1.0])
    assert series["signal_groups_all_logged"].tolist() == pytest.approx([8.0, 0.0])


def test_the_sampler_order_is_trls_and_drops_the_incomplete_last_chunk():
    torch = pytest.importorskip("torch")
    step = S.training_order(40, prompts_per_step=16, seed=42)
    perm = torch.randperm(40, generator=torch.Generator().manual_seed(42)).tolist()
    assert [step[r] for r in perm[:16]] == [1] * 16
    assert [step[r] for r in perm[16:32]] == [2] * 16
    assert [step[r] for r in perm[32:]] == [0] * 8          # 8 prompts never used
    assert (S.training_order(40, 16, 42) == step).all()


def test_unbiased_group_estimates_from_stored_samples():
    assert S.zero_variance_u(8, 8) == 1.0 and S.zero_variance_u(5, 8) == 0.0
    # 32 samples, 30 right: P(8 drawn all right) = C(30,8)/C(32,8)
    from math import comb
    assert S.all_agree_u(30, 32) == pytest.approx(comb(30, 8) / comb(32, 8))
    assert S.zero_variance_u(30, 32) == pytest.approx(comb(30, 8) / comb(32, 8))
