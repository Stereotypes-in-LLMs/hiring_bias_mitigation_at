"""Preference optimisation on matched invariant-vs-biased response pairs.

The `chosen` response holds its verdict and its rationale invariant to the injected
attribute; the `rejected` response, on the identical job, CV, attribute and condition, lets
the attribute drive the outcome. Matching that tightly is what keeps the learned preference
about bias: any looser pairing and the model can satisfy the objective by preferring longer
or better-written text.

Two objectives, both supported through one config field:

`dpo`
    Direct Preference Optimisation. Needs a frozen reference model, which on a 12B target
    means a second copy in memory -- feasible on the Spark's 128 GB with LoRA (the reference
    is the base model with the adapter disabled, so it costs nothing extra), not with a full
    fine-tune of a 12B.

`kto`
    Kahneman-Tversky Optimisation. Scores each response on its own rather than against a
    partner, so it answers whether the preference signal needs pairing at all. Its split is
    the DPO pairs unpaired into (prompt, completion, label) -- identical responses, so a
    difference between the two arms is the objective and not the data.

    This arm replaces ORPO, which the study originally planned. TRL 1.x removed
    ORPOConfig/ORPOTrainer outright, and pinning an older TRL would have meant downgrading
    transformers below what vLLM needs -- breaking the evaluation pipeline that had already
    produced forty scored runs. Report ORPO as removed for a dependency reason, not as a
    result that came out uninteresting.

The SFT-only arm and this one share their data, so the ablation is clean: same rows, same
`chosen` responses, the only difference being whether the model also saw what to reject.

Run:
    python -m hiring_bias_mitigation.mitigation.dpo --config configs/mitigation/dpo/<x>.yaml
"""

from __future__ import annotations

import argparse
import os

from ..utils.config import load_config
from ..utils.logging import get_logger
from ..utils.seed import set_seed
from . import training_common as C

log = get_logger(__name__)

#: Config field `objective`. Each names both the trainer and the data split it reads.
OBJECTIVES = ("dpo", "kto")

#: DPOConfig fields a config may set and that reach TRL unchanged when present. Absent keys
#: are not passed, so TRL's own defaults apply and a config describes exactly what differs.
#:
#: `loss_type: [sigmoid, sft]` with `loss_weights` adds an NLL term on the chosen response to
#: the DPO loss -- the guard against likelihood displacement on closely matched pairs (Razin
#: et al. 2025; Pal et al. 2024), which the v2 covert negatives are. TRL 1.x removed
#: `rpo_alpha`; this combination is its replacement. `ld_alpha` is LD-DPO's length
#: desensitisation, for when the two sides still differ in length.
DPO_CONFIG_PASSTHROUGH = (
    "loss_type", "loss_weights", "ld_alpha", "label_smoothing", "precompute_ref_log_probs",
    "sync_ref_model",
)


def dpo_config_candidates(cfg: dict) -> dict:
    """Everything a config can say about the DPO objective, before version filtering."""
    candidates = {
        "beta": float(cfg.get("beta", 0.1)),
        "max_length": cfg.get("max_seq_len", 2048),
        # Removed in TRL 1.x, which truncates through max_length and truncation_mode instead.
        "max_prompt_length": cfg.get("max_prompt_len", 1792),
    }
    for key in DPO_CONFIG_PASSTHROUGH:
        if key in cfg:
            candidates[key] = cfg[key]
    return candidates


def _to_conversational(dataset, tokenizer, chat_template_kwargs: dict | None = None):
    """Renders prompt/chosen/rejected through the chat template, split exactly at the prompt.

    The same three constraints as SFT's `render_chat`, and a stronger reason to meet them. A
    DPO objective *is* a difference of log-probabilities between the chosen and the rejected
    continuation. Handing TRL message lists makes it tokenise the prompt on its own, and
    Qwen's template emits `<think>\n` for a generation prompt against `<think>\n\n</think>\n\n`
    in the full render — the same string, different tokens. TRL then reports "Mismatch between
    tokenized prompt and the start of tokenized prompt+chosen" and masks on a guess, so the
    difference it optimises is taken partly over prompt tokens.

    Each response is taken as the **string remainder** of the full render after the prompt
    render, which reproduces the template's own output byte for byte. The template kwargs
    default to the audit's (`enable_thinking=False`), so the model is trained on the format it
    is later asked to produce.

    Columns other than the three roles are dropped: they are provenance for the analysis.
    """
    from .sft import DEFAULT_CHAT_TEMPLATE_KWARGS

    kwargs = dict(chat_template_kwargs or DEFAULT_CHAT_TEMPLATE_KWARGS)

    def render(example: dict) -> dict:
        user = [{"role": "user", "content": example["prompt"]}]
        prompt = tokenizer.apply_chat_template(
            user, tokenize=False, add_generation_prompt=True, **kwargs
        )
        out = {"prompt": prompt}
        for side in ("chosen", "rejected"):
            full = tokenizer.apply_chat_template(
                [*user, {"role": "assistant", "content": example[side]}],
                tokenize=False, **kwargs,
            )
            if not full.startswith(prompt):
                raise ValueError(
                    f"the chat template does not render the prompt as a prefix of "
                    f"prompt+{side}, so the preference margin cannot be computed on the "
                    f"response alone."
                )
            out[side] = full[len(prompt):]
        return out

    keep = ("prompt", "chosen", "rejected")
    return dataset.map(
        render, remove_columns=[c for c in dataset.column_names if c not in keep]
    )


def decision_callbacks(cfg: dict, splits: dict, tokenizer, output_dir) -> list:
    """The decision-level eval (mitigation/decision_eval.py) after every evaluation, on
    `decision_eval_rows` validation prompts, when a config asks for it."""
    n = int(cfg.get("decision_eval_rows") or 0)
    if not n or "validation" not in splits:
        return []
    from .decision_eval import DecisionEvalCallback, eval_rows

    rows = eval_rows(splits["validation"].to_pandas(), n, int(cfg.get("seed", 42)))
    return [DecisionEvalCallback(
        rows, tokenizer, output_dir, template_kwargs=cfg.get("chat_template_kwargs"),
        max_new_tokens=int(cfg.get("decision_eval_max_new_tokens", 96)))]


def train(cfg: dict) -> C.TrainArtifacts:
    objective = cfg.get("objective", "dpo")
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; expected one of {OBJECTIVES}")
    if cfg.get("precompute_ref_log_probs") and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # TRL writes the precomputed cache on rank 0 into the dataset's cache dir, which for
        # these in-memory datasets is a per-process temp dir the other ranks cannot read: job
        # 5054142 died there after 6 minutes. Refuse before anything is loaded.
        raise ValueError("precompute_ref_log_probs does not work under torchrun with in-memory "
                         "datasets; set it false (the reference pass then runs every step)")

    set_seed(cfg.get("seed", 42))
    C.wandb_init(cfg)
    output_dir = C.resolve_run_dir(cfg)
    # Rank 0 is the only process that logs to W&B.
    if C.is_main_process():
        C.wandb_resume(cfg, output_dir)

    # KTO reads its own split: same responses as DPO, one row per response.
    # `data_kind` lets a variant arm read a different preference split -- e.g. decision-only
    # pairs -- through the same trainer.
    splits = C.load_generated_dataset(cfg, cfg.get("data_kind", objective))
    model, tokenizer = C.load_model_and_tokenizer(cfg)
    chat_kwargs = cfg.get("chat_template_kwargs")
    train_ds = _to_conversational(splits["train"], tokenizer, chat_kwargs)
    eval_ds = (
        _to_conversational(splits["validation"], tokenizer, chat_kwargs)
        if "validation" in splits
        else None
    )

    peft = C.peft_config(cfg)
    base_args = C.training_arguments(
        cfg, output_dir, has_eval=eval_ds is not None, n_train=len(train_ds)
    ).to_dict()

    if objective == "dpo":
        from trl import DPOConfig, DPOTrainer

        args = DPOConfig(
            **base_args,
            **C.supported_kwargs(DPOConfig, dpo_config_candidates(cfg), "DPOConfig"),
        )
        trainer = DPOTrainer(
            model=model,
            # With LoRA, TRL uses this same model with the adapter disabled as the reference,
            # so nothing extra is held in memory. Without LoRA it deep-copies the model --
            # which is why the 12B targets are LoRA-only in the shipped configs.
            ref_model=None,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft,
            callbacks=C.early_stopping_callbacks(cfg, eval_ds is not None)
            + C.memory_guard_callbacks(cfg)
            + decision_callbacks(cfg, splits, tokenizer, output_dir),
        )
    else:
        # KTO, not ORPO. TRL 1.x removed ORPOConfig/ORPOTrainer entirely; KTO is the second
        # preference objective that survives, and it runs on the same responses -- the KTO
        # split is the DPO pairs unpaired into (prompt, completion, label), so a difference
        # between the two arms is the objective rather than the data.
        from trl import KTOConfig, KTOTrainer

        args = KTOConfig(
            **base_args,
            **C.supported_kwargs(
                KTOConfig,
                {
                    "beta": float(cfg.get("beta", 0.1)),
                    "max_length": cfg.get("max_seq_len", 2048),
                    "max_prompt_length": cfg.get("max_prompt_len", 1792),
                },
                "KTOConfig",
            ),
        )
        trainer = KTOTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=peft,
            callbacks=C.early_stopping_callbacks(cfg, eval_ds is not None)
            + C.memory_guard_callbacks(cfg),
        )

    log.info(
        "%s %s on %d %s",
        objective.upper(), cfg["base_model"], len(train_ds),
        "preference pairs" if objective == "dpo" else "labelled responses",
    )
    manifest = {
        "mitigation_family": "dpo",
        "objective": objective,
        "beta": cfg.get("beta", 0.1),
        "n_train": len(train_ds),
        "n_eval": len(eval_ds) if eval_ds else 0,
        "sft_checkpoint": cfg.get("sft_checkpoint"),
    }
    # Written before training as well: a run cut by a time limit never reaches the final
    # write, and without `sft_checkpoint` on disk its checkpoint-N adapters would be merged
    # without their SFT parent (merge.adapter_chain).
    if C.is_main_process():
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": "training"})
    # `resume: true` continues from the newest finished checkpoint in output_dir, so a run
    # longer than the partition's time limit is the same job submitted again.
    resume = C.last_checkpoint(output_dir) if cfg.get("resume") else None
    if resume:
        log.info("resuming from %s", resume)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    if C.is_main_process():
        tokenizer.save_pretrained(str(output_dir))
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": "done"})
    return C.TrainArtifacts(output_dir, len(train_ds), len(eval_ds) if eval_ds else 0)


def main() -> None:
    C.configure_cuda_allocator()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("stage", "dpo")

    # A DPO config may point base_model at an SFT run's output directory -- or at a published
    # adapter on the Hub. Either resolves to a local adapter directory that the loader merges
    # into its base model before a fresh LoRA is fitted on top.
    if cfg.get("sft_checkpoint"):
        cfg["base_model"] = C.resolve_sft_checkpoint(cfg["sft_checkpoint"])

    artifacts = train(cfg)
    log.info("preference optimisation complete -> %s", artifacts.output_dir)


if __name__ == "__main__":
    main()
