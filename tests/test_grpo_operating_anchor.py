"""The operating-point anchor of mitigation/grpo_operating_anchor.py, on CPU with a toy model."""
from __future__ import annotations

import contextlib
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.mitigation.grpo_operating_anchor import (  # noqa: E402
    anchor_penalty,
    trainer_class,
)

HIRE, REJECT, VOCAB = 1, 2, 4


class Toy(torch.nn.Module):
    """Logits from a per-token table, plus a trainable 'adapter' bias on hire."""

    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(10, VOCAB), requires_grad=False)
        self.table.data[:, HIRE] = torch.linspace(-1, 1, 10)
        self.adapter = torch.nn.Parameter(torch.zeros(()))
        self.enabled = True
        self.training = True

    def forward(self, input_ids, logits_to_keep=None):
        logits = self.table[input_ids].clone()
        if self.enabled:
            logits[..., HIRE] = logits[..., HIRE] + self.adapter
        return SimpleNamespace(logits=logits)

    @contextlib.contextmanager
    def disable_adapter(self):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


class Tok:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[int(c) for c in text]])}


class FakeBase:
    def __init__(self, model):
        self.model = model
        self.processing_class = Tok()
        self.accelerator = SimpleNamespace(process_index=0, device="cpu",
                                           unwrap_model=lambda m: m)
        self._metrics = {"train": defaultdict(list)}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return torch.zeros(())


def test_penalty_is_the_squared_mean_shift():
    pen, shift = anchor_penalty(torch.tensor([0.6, 0.4]), torch.tensor([0.5, 0.3]), 10.0)
    assert torch.isclose(shift, torch.tensor(0.1)) and torch.isclose(pen, torch.tensor(0.1))


def test_no_penalty_at_the_reference_and_a_restoring_gradient_away_from_it():
    model = Toy()
    cls = trainer_class(FakeBase, prompts=["01", "05", "09"], weight=10.0, rows_per_step=2,
                        seed=0, token_ids=(HIRE, REJECT))
    trainer = cls(model)
    assert float(trainer.compute_loss(model, {})) == 0.0
    model.adapter.data.fill_(1.0)  # the policy now hires more than SFT on every prompt
    loss = trainer.compute_loss(model, {})
    loss.backward()
    assert float(loss) > 0 and model.adapter.grad > 0  # descent lowers the hire bias
    assert trainer._metrics["train"]["anchor/shift"][-1] > 0
    # the reference was cached with the adapter disabled
    assert all(0 < v < 1 for v in trainer._anchor_ref.values())


def test_reference_does_not_move_with_the_policy():
    model = Toy()
    cls = trainer_class(FakeBase, prompts=["01", "05", "09"], weight=1.0, rows_per_step=3,
                        seed=0, token_ids=(HIRE, REJECT))
    trainer = cls(model)
    trainer.compute_loss(model, {})
    before = dict(trainer._anchor_ref)
    model.adapter.data.fill_(-2.0)
    trainer.compute_loss(model, {})
    assert trainer._anchor_ref == before
