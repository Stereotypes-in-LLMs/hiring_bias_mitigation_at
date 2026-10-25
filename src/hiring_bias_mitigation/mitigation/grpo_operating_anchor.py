"""GRPO with an operating-point anchor: a penalty on the policy's mean hire probability moving
away from SFT-v3's on a fixed set of held-out prompts SFT-v3 is unsure of.

Why (docs/GRPO_IMPROVE.md, round 1 and R2-a). In every GRPO-improve arm the benchmark's hire
threshold wandered by several pp of TPR per 25 steps, and not through the first-order
decision-bias gradient: projecting that out (A1) doubled the swing. That push looks like a
restoring force on the training rows, which the policy is sure of by step ~125, so nothing
holds the threshold where the benchmark's borderline prompts sit. This adds one there. Across
the seven arms of the monitor calibration (job 5123830), the mean margin shift on these rows
correlates 0.93 with the arm's benchmark TPR change.

What. Each micro-step, on every rank, `rows_per_step` anchor prompts drawn with a seeded RNG
are forwarded through the policy one at a time (no padding: Qwen3.5's linear-attention layers
are not checked against left padding here) with the op-margin-v2 decision prefix. With
p = sigmoid(logit_hire - logit_reject), the loss gains

    weight * (mean_batch(p_policy) - mean_batch(p_sft))^2

where p_sft is the same prompt's value with the LoRA disabled, computed once and cached. Only
the batch mean is held: individual prompts can move, the operating point cannot drift.
"""

from __future__ import annotations

import numpy as np
import torch


def anchor_penalty(p_policy: torch.Tensor, p_reference: torch.Tensor, weight: float):
    """weight * (mean(p_policy) - mean(p_reference))^2, and the shift itself."""
    shift = p_policy.mean() - p_reference.mean()
    return weight * shift.pow(2), shift


def trainer_class(base, *, prompts: list[str], weight: float, rows_per_step: int, seed: int,
                  token_ids: tuple[int, int]):
    """`base` (TRL's GRPOTrainer) with the anchor penalty added to each training micro-step.

    `prompts` are the anchor rows already rendered through the chat template with the decision
    prefix appended; `token_ids` the first tokens of hire and reject after that prefix.
    """
    hire_id, reject_id = token_ids

    class OperatingAnchorGRPOTrainer(base):
        def _anchor_setup(self):
            if getattr(self, "_anchor_ids", None) is not None:
                return
            tok = self.processing_class
            self._anchor_ids = [tok(p, add_special_tokens=False, return_tensors="pt")["input_ids"]
                                for p in prompts]
            self._anchor_ref = {}
            self._anchor_rng = np.random.default_rng(seed + 1000 * self.accelerator.process_index)

        def _anchor_prob(self, module, i: int) -> torch.Tensor:
            ids = self._anchor_ids[i].to(self.accelerator.device)
            logits = module(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
            return torch.sigmoid(logits[hire_id] - logits[reject_id])

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                        num_items_in_batch=num_items_in_batch)
            if not model.training or weight <= 0:
                return loss
            self._anchor_setup()
            module = self.accelerator.unwrap_model(model)
            picks = self._anchor_rng.choice(len(prompts), size=rows_per_step, replace=False)
            missing = [int(i) for i in picks if int(i) not in self._anchor_ref]
            if missing:
                with torch.no_grad(), module.disable_adapter():
                    for i in missing:
                        self._anchor_ref[i] = float(self._anchor_prob(module, i))
            p_policy = torch.stack([self._anchor_prob(module, int(i)) for i in picks])
            p_ref = torch.tensor([self._anchor_ref[int(i)] for i in picks],
                                 device=p_policy.device)
            penalty, shift = anchor_penalty(p_policy, p_ref, weight)
            self._metrics["train"]["anchor/shift"].append(float(shift.detach()))
            self._metrics["train"]["anchor/penalty"].append(float(penalty.detach()))
            if return_outputs:
                value, outputs = loss
                return value + penalty.to(value.dtype), outputs
            return loss + penalty.to(loss.dtype)

    return OperatingAnchorGRPOTrainer
