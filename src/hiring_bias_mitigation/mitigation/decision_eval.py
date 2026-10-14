"""Decision-level evaluation of a preference-trained model: what it decides, not its loss.

DPO's loss and reward accuracy say how far the preference moved between two given texts. They
do not say whether the model, decoding on its own, now gives the reference verdict on
attribute-bearing prompts, or whether its hire rate drifted on the way. This module
greedy-decodes a fixed, seeded, reference-balanced set of validation prompts and reports:

  agree_reference          share of decisions equal to the attribute-free reference
  hire_rate                share decided hire; with hire_rate_ref_hire / _ref_reject, the
                           operating point on each side of the reference (a model that
                           "improves" by rejecting more shows up here)
  invalid_rate             share that parse to no decision (refusal or broken JSON)
  agree_reference/<group>  agreement per protected group

Used two ways: `DecisionEvalCallback` during training (config `decision_eval_rows`), and
scripts/eval_dpo_checkpoints.py over a finished run's checkpoints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from transformers import TrainerCallback

from ..eval.backends import _chat_wrap
from ..eval.parsing import HIRE, REJECT, parse_output
from ..utils.logging import get_logger

log = get_logger(__name__)

#: Columns carried next to each output in the per-step records and sample tables.
META_COLUMNS = ("pair_id", "reference_decision", "protected_group", "protected_attr",
                "condition")


def eval_rows(frame: pd.DataFrame, n: int | None = None, seed: int = 42) -> pd.DataFrame:
    """`n` rows, half per reference verdict where the frame allows, one seeded draw; all rows
    when `n` is None or covers the frame."""
    if n is None or n >= len(frame):
        return frame.reset_index(drop=True)
    parts = []
    for decision in ("hire", "reject"):
        side = frame[frame["reference_decision"] == decision]
        parts.append(side.sample(n=min(n // 2, len(side)), random_state=seed))
    return pd.concat(parts).sort_index().reset_index(drop=True)


def decision_metrics(rows: pd.DataFrame, outputs: list[str], lang: str = "en") -> dict:
    decisions = pd.Series([parse_output(o, lang).decision for o in outputs], index=rows.index)
    reference = rows["reference_decision"].astype(str).str.lower()
    agree = decisions == reference
    hire = decisions == HIRE
    metrics: dict = {
        "n": len(rows),
        "agree_reference": float(agree.mean()),
        "hire_rate": float(hire.mean()),
        "invalid_rate": float((~decisions.isin([HIRE, REJECT])).mean()),
    }
    for side in ("hire", "reject"):
        mask = reference == side
        metrics[f"hire_rate_ref_{side}"] = float(hire[mask].mean()) if mask.any() else None
    if "protected_group" in rows.columns:
        for group, index in rows.groupby("protected_group", sort=True).groups.items():
            metrics[f"agree_reference/{group}"] = float(agree[index].mean())
    return metrics


def generate(model, tokenizer, prompts: list[str], batch_size: int = 16,
             max_new_tokens: int = 96, template_kwargs: dict | None = None) -> list[str]:
    """Greedy completions through the audit's chat template, left-padded, KV cache on."""
    import torch

    texts = [_chat_wrap(tokenizer, p, template_kwargs) for p in prompts]
    side, tokenizer.padding_side = tokenizer.padding_side, "left"
    was_training = model.training
    model.eval()
    outputs: list[str] = []
    try:
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                                  padding=True).to(model.device)
                out = model.generate(**batch, do_sample=False, max_new_tokens=max_new_tokens,
                                     use_cache=True, pad_token_id=tokenizer.pad_token_id)
                outputs += tokenizer.batch_decode(out[:, batch["input_ids"].shape[1]:],
                                                  skip_special_tokens=True)
    finally:
        tokenizer.padding_side = side
        if was_training:
            model.train()
    return outputs


def record(output_dir: str | Path, step: int, rows: pd.DataFrame, outputs: list[str],
           metrics: dict, prefix: str = "decision", n_samples: int = 8) -> None:
    """Appends to <output_dir>/<prefix>_eval.jsonl and, if a W&B run is active, logs the
    metrics against the trainer's step with a small sample table."""
    with (Path(output_dir) / f"{prefix}_eval.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"step": step, "metrics": metrics}) + "\n")
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "run", None) is None:
        return
    meta = [c for c in META_COLUMNS if c in rows.columns]
    head = rows[meta].head(n_samples).astype(str).values.tolist()
    table = wandb.Table(columns=[*meta, "output"],
                        data=[[*m, o] for m, o in zip(head, outputs[:n_samples])])
    wandb.log({**{f"{prefix}/{k}": v for k, v in metrics.items() if v is not None},
               f"{prefix}/samples": table, "train/global_step": step})


class DecisionEvalCallback(TrainerCallback):
    """Greedy-decodes the fixed rows after every evaluation, on rank 0 only; other ranks wait
    at their next collective, well inside the NCCL timeout for a few dozen prompts."""

    def __init__(self, rows: pd.DataFrame, tokenizer, output_dir: str | Path,
                 lang: str = "en", batch_size: int = 16, max_new_tokens: int = 96,
                 template_kwargs: dict | None = None):
        self.rows = rows.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.lang = lang
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.template_kwargs = template_kwargs

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        outputs = generate(model, self.tokenizer, self.rows["prompt"].tolist(),
                           batch_size=self.batch_size, max_new_tokens=self.max_new_tokens,
                           template_kwargs=self.template_kwargs)
        metrics = decision_metrics(self.rows, outputs, self.lang)
        record(self.output_dir, state.global_step, self.rows, outputs, metrics)
        log.info("decision eval at step %d: agree %.3f, hire rate %.3f", state.global_step,
                 metrics["agree_reference"], metrics["hire_rate"])


def checkpoint_steps(run_dir: str | Path) -> list[int]:
    """0 (the starting model) and every checkpoint-N the trainer finished writing."""
    root = Path(run_dir)
    steps = [int(m.group(1)) for d in root.iterdir()
             if (m := re.fullmatch(r"checkpoint-(\d+)", d.name))
             and (d / "trainer_state.json").is_file()]
    return [0, *sorted(steps)]
