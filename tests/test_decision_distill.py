"""Counterfactual decision distillation (mitigation/decision_distill.py and its data builder),
on CPU with a toy model."""
from __future__ import annotations

import contextlib
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hiring_bias_mitigation.mitigation.decision_distill import (  # noqa: E402
    collate,
    decision_kl,
    trainer_class,
)

_spec = importlib.util.spec_from_file_location(
    "build_distill", ROOT / "scripts" / "build_decision_distill_data.py")
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

HIRE, REJECT, VOCAB = 1, 2, 5


def logits(h, r):
    row = torch.zeros(VOCAB)
    row[HIRE], row[REJECT] = h, r
    return row.unsqueeze(0)


def test_kl_is_zero_at_the_target_and_matches_the_binary_formula():
    assert float(decision_kl(logits(1.0, 0.0), logits(1.0, 0.0), (HIRE, REJECT))) < 1e-7
    p, q = 1 / (1 + math.exp(-0.5)), 1 / (1 + math.exp(-2.0))
    want = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))
    got = float(decision_kl(logits(0.5, 0.0), logits(2.0, 0.0), (HIRE, REJECT)))
    assert abs(got - want) < 1e-6


class Toy(torch.nn.Module):
    """Logits from a fixed per-token table plus a trainable 'adapter' bias on hire."""

    def __init__(self):
        super().__init__()
        self.register_buffer("table", torch.zeros(10, VOCAB))
        self.table[:, HIRE] = torch.linspace(-2, 2, 10)
        self.adapter = torch.nn.Parameter(torch.zeros(()))
        self.enabled = True

    def forward(self, input_ids, logits_to_keep=None):
        out = self.table[input_ids].clone()
        if self.enabled:
            out[..., HIRE] = out[..., HIRE] + self.adapter
        return SimpleNamespace(logits=out)

    @contextlib.contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def run_loss(model, student_ids, teacher_ids, is_free=False):
    cls = trainer_class((HIRE, REJECT))
    trainer = cls.__new__(cls)  # compute_loss only needs these attributes
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda m: m)
    trainer._distill_log = {}
    inputs = {"input_ids": torch.tensor([student_ids]), "teacher_ids": torch.tensor([teacher_ids]),
              "is_free": torch.tensor([is_free])}
    return trainer, trainer.compute_loss(model, inputs)


def test_teacher_reads_the_free_prompt_with_the_adapter_off():
    model = Toy()
    model.adapter.data.fill_(3.0)  # the policy has drifted toward hire
    trainer, loss = run_loss(model, [0, 4], [0, 4])  # same prompt: only the adapter differs
    loss.backward()
    assert float(loss) > 0 and model.adapter.grad > 0  # descent pulls back toward SFT-v3
    _kl, p, q = trainer._distill_log["attr"][0]
    assert p > q  # student (adapter on) hires more than the teacher (adapter off)


def test_attributed_row_moves_toward_the_free_decision():
    model = Toy()
    _, loss = run_loss(model, [0, 2], [0, 8])  # attributed prompt hires less than its free one
    loss.backward()
    assert model.adapter.grad < 0  # descent raises the hire logit on the attributed prompt


def test_one_prompt_per_device():
    row = {"input_ids": [1, 2], "teacher_ids": [1, 3], "is_free": False}
    assert collate([row])["input_ids"].shape == (1, 2)
    with pytest.raises(ValueError):
        collate([row, row])


def test_builder_pairs_rows_and_adds_one_free_anchor_per_pair():
    rows = pd.DataFrame({
        "prompt": ["A with x", "A with y", "B with x"], "pair_id": ["a", "a", "b"],
        "candidate_id": ["ca", "ca", "cb"], "job_id": ["ja", "ja", "jb"],
        "reference_decision": ["hire", "hire", "reject"], "protected_group": ["g"] * 3,
        "protected_attr": ["x", "y", "x"], "condition": ["explicit"] * 3, "tipped": [True] * 3,
        "free_hire_rate": [0.5, 0.5, 0.0]})
    free = pd.Series({"a": "A", "b": "B"})
    out = B.build(rows, free)
    assert len(out) == 5 and out["is_free"].sum() == 2
    anchors = out[out["is_free"]]
    assert (anchors["prompt"] == anchors["free_prompt"]).all()
    assert set(out.loc[~out["is_free"], "free_prompt"]) == {"A", "B"}
    with pytest.raises(ValueError):
        B.build(rows, pd.Series({"a": "A"}))
