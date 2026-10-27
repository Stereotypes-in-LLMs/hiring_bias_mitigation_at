"""Counterfactual decision distillation: the attributed prompt's decision learns the
attribute-free prompt's, with SFT-v3 as a frozen teacher (docs/DECISION_DISTILL.md).

The frontier family the 2026-10 survey (docs/FRONTIER_RL_SURVEY.md, family A) found best
matched to this task: on-policy context distillation in its decision-token form (OPCD,
arXiv 2602.12275; counterfactual logit pairing, Garg et al. 2019, arXiv 1809.10610). The
student is the policy reading a CV with a protected attribute injected; the teacher is the same
weights with the LoRA disabled, i.e. SFT-v3, reading the pair's attribute-free CV. After the
op-margin-v2 decision prefix, both give a two-way distribution over the first tokens of hire
and reject, and the loss is

    KL(q_teacher(. | x_free) || p_student(. | x_attr))

on every row. Each pair also contributes its attribute-free prompt as a row of its own
(`is_free`), whose target is the teacher on the same prompt: at the start the loss there is 0,
and it holds the model's attribute-free decisions, and so its operating point, where SFT-v3 has
them.

Why this and not more RL (docs/GRPO_IMPROVE.md, section 11). The evaluated quantity is whether
greedy decisions agree across a pair's attribute values; this loss asks for exactly that,
densely, on every row, including the 80-98% of GRPO groups whose rewards all agree. Its target
is the model's own attribute-free judgement, not the vote anchor, which on 37% of the training
rows teaches a different hiring standard.

One prompt per device and micro-step: no padding, so the last position is the decision
position, and Qwen3.5's linear-attention layers never see a padded batch. The teacher's
forward runs under no_grad on the unwrapped model, outside DDP's graph.
"""

from __future__ import annotations

import argparse

from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from . import training_common as C

log = get_logger(__name__)


def decision_kl(student_logits, teacher_logits, token_ids):
    """KL(q || p) over the two decision tokens, per row; logits are [rows, vocab] (float)."""
    import torch

    idx = torch.tensor(token_ids, device=student_logits.device)
    log_p = torch.log_softmax(student_logits.index_select(-1, idx).float(), dim=-1)
    log_q = torch.log_softmax(teacher_logits.index_select(-1, idx).float(), dim=-1)
    return (log_q.exp() * (log_q - log_p)).sum(-1)


def tokenize_rows(frame, tokenizer, chat_kwargs):
    """`input_ids` for the student (the row's prompt) and `teacher_ids` (the pair's
    attribute-free prompt), each through the chat template with the decision prefix."""
    from datasets import Dataset

    from ..eval.backends import _chat_wrap
    from ..eval.margins import PREFIX

    def enc(text):
        return tokenizer(_chat_wrap(tokenizer, text, chat_kwargs) + PREFIX,
                         add_special_tokens=False)["input_ids"]

    return Dataset.from_dict({
        "input_ids": [enc(p) for p in frame["prompt"]],
        "teacher_ids": [enc(p) for p in frame["free_prompt"]],
        "is_free": [bool(v) for v in frame["is_free"]],
    })


def collate(rows):
    import torch

    if len(rows) != 1:
        raise ValueError("decision distillation runs one prompt per device and micro-step")
    row = rows[0]
    return {"input_ids": torch.tensor([row["input_ids"]]),
            "teacher_ids": torch.tensor([row["teacher_ids"]]),
            "is_free": torch.tensor([row["is_free"]])}


def trainer_class(token_ids):
    from transformers import Trainer

    class DecisionDistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch

            student = model(input_ids=inputs["input_ids"], logits_to_keep=1).logits[:, -1]
            module = self.accelerator.unwrap_model(model)
            with torch.no_grad(), module.disable_adapter():
                teacher = module(input_ids=inputs["teacher_ids"], logits_to_keep=1).logits[:, -1]
            kl = decision_kl(student, teacher, token_ids)
            loss = kl.mean()
            if model.training:
                idx = torch.tensor(token_ids, device=student.device)
                p = torch.softmax(student.detach().index_select(-1, idx).float(), -1)[:, 0]
                q = torch.softmax(teacher.index_select(-1, idx).float(), -1)[:, 0]
                key = "free" if bool(inputs["is_free"][0]) else "attr"
                self._distill_log.setdefault(key, []).append(
                    (float(kl.detach().mean()), float(p.mean()), float(q.mean())))
            return (loss, {"logits": student}) if return_outputs else loss

        def log(self, logs, start_time=None):
            for key, values in getattr(self, "_distill_log", {}).items():
                if values:
                    n = len(values)
                    logs[f"distill/{key}_kl"] = sum(v[0] for v in values) / n
                    logs[f"distill/{key}_p_hire"] = sum(v[1] for v in values) / n
                    logs[f"distill/{key}_q_hire"] = sum(v[2] for v in values) / n
            self._distill_log = {}
            super().log(logs, start_time)

    return DecisionDistillTrainer


def train(cfg: dict) -> C.TrainArtifacts:
    import pandas as pd
    from peft import get_peft_model

    from ..eval.margins import WORDS, decision_token_ids
    from ..utils.config import resolve_output_path

    set_seed(cfg.get("seed", 42))
    C.wandb_init(cfg)
    output_dir = C.resolve_run_dir(cfg)
    if C.is_main_process():
        C.wandb_resume(cfg, output_dir)
    root = resolve_output_path(cfg["data"]["artifacts_dir"])
    frame = pd.read_parquet(f"{root}/distill_train.parquet")
    model, tokenizer = C.load_model_and_tokenizer(cfg)
    model.enable_input_require_grads()
    model = get_peft_model(model, C.peft_config(cfg))
    chat_kwargs = cfg.get("chat_template_kwargs")
    token_ids = decision_token_ids(tokenizer, WORDS["en"])
    train_ds = tokenize_rows(frame, tokenizer, chat_kwargs)
    args = C.training_arguments(cfg, output_dir, has_eval=False, n_train=len(train_ds))
    args.remove_unused_columns = False
    args.save_strategy = "steps"
    args.save_steps = int(cfg.get("save_steps", 50))
    trainer = trainer_class(token_ids)(
        model=model, args=args, train_dataset=train_ds, data_collator=collate,
        callbacks=C.memory_guard_callbacks(cfg) + C.cost_callbacks())
    trainer._distill_log = {}
    manifest = {"mitigation_family": "decision_distill", "objective": "decision_distill",
                "n_train": len(train_ds), "n_free_rows": int(frame["is_free"].sum()),
                "decision_token_ids": list(token_ids),
                "sft_checkpoint": cfg.get("sft_checkpoint")}
    if C.is_main_process():
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": "training"})
    resume = C.last_checkpoint(output_dir) if cfg.get("resume") else None
    log.info("decision distillation from %s on %d rows (%d attribute-free)",
             cfg["base_model"], len(train_ds), manifest["n_free_rows"])
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    if C.is_main_process():
        tokenizer.save_pretrained(str(output_dir))
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": "done"})
    return C.TrainArtifacts(output_dir, len(train_ds), 0)


def main() -> None:
    C.configure_cuda_allocator()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("stage", "decision_distill")
    if cfg.get("sft_checkpoint"):
        cfg["base_model"] = C.resolve_sft_checkpoint(cfg["sft_checkpoint"])
    artifacts = train(cfg)
    log.info("decision distillation complete -> %s", artifacts.output_dir)


if __name__ == "__main__":
    main()
