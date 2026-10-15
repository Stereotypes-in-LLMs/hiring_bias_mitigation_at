"""GRPO from the SFT model: the decision itself as the reward, no teacher-written negatives.

Why this arm. Three rounds of DPO continued from SFT gained at best -0.96 pp unstable sets at
matched hire rate (CI -1.96 to +0.04; docs/ITERATIVE_DPO.md). DPO needs a `rejected` text,
and the student's own attribute-tipped mistakes ran dry and point one way. GRPO needs only a
prompt and a way to score a completion: the policy samples `num_generations` answers per
attributed prompt and each is rewarded for reaching the pair's attribute-free reference
verdict, so a wrong answer in either direction costs the same.

Rewards, plain Python, no model in the loop:

  decision_reward  1.0 when `parse_output(completion).decision` equals `reference_decision`
  format_reward    0.2 when the completion is exactly one JSON object with string `decision`
                   and `feedback` fields and the feedback runs FEEDBACK_WORDS words; the SFT
                   student meets it on 99.9% of its samples, so it guards against drift

Policy and reference. The SFT adapter is merged into Qwen3.5-9B in memory and a fresh LoRA is
fitted on top (training_common.load_model_and_tokenizer, as for DPO). With PEFT, TRL takes the
reference log-probabilities with the adapter disabled, so the KL term (`beta`) pulls toward
SFT. `run_config.json` names the SFT checkpoint, so merge.adapter_chain merges SFT then GRPO.

Rollouts. TRL's colocated vLLM, one engine per GPU, is started from the plain base model and
receives the policy's weights before every generation. Three things are changed from TRL's
defaults, each checked by a test or by `SyncValidationCallback` before a run trains:

  * Merged weights only, computed out of place. TRL merges the adapter into the base layers
    in place and unmerges it after the push (`merge_adapter`, `unmerge_adapter`): in bf16
    that is a rounding round trip on the frozen weights every step, and those weights are
    also the KL reference. `merged_named_params` adds the LoRA delta in fp32 into a copy.
    vLLM's LoRA path, which does not reproduce Qwen3.5 adapters (mitigation/merge.py), is
    never involved.
  * Names. The policy is the text model (`model.layers.N...`); vLLM serves the checkpoint's
    Qwen3_5ForConditionalGeneration, whose loader maps `model.language_model.` names, the
    names the checkpoint itself uses. `vllm_param_name` translates.
  * Engine arguments. TRL builds `LLM(...)` itself; `vllm_engine_kwargs` from the config
    (on Daint, `gdn_prefill_backend: triton`) are added to that call.

Prompts arrive rendered through the audit's chat template (`_chat_wrap`, thinking off), so
TRL tokenises them as plain text and applies no template of its own.

Run:
    python -m hiring_bias_mitigation.mitigation.grpo --config configs/mitigation/grpo/<x>.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..eval.parsing import HIRE, REJECT, parse_output
from ..utils.config import load_config
from ..utils.logging import get_logger
from . import training_common as C

log = get_logger(__name__)

DECISION_REWARD = 1.0
FORMAT_REWARD = 0.2
#: Feedback length the format reward accepts, in words. The prompt asks for at most 30; the
#: SFT student writes 14 to 27 (1st to 99th percentile of 16k samples).
FEEDBACK_WORDS = (4, 45)

#: GRPOConfig fields a config may set; absent keys keep TRL's defaults.
GRPO_CONFIG_PASSTHROUGH = (
    "beta", "num_generations", "temperature", "top_p", "top_k", "max_completion_length",
    "loss_type", "scale_rewards", "epsilon", "epsilon_high", "num_iterations",
    "steps_per_generation", "generation_batch_size", "mask_truncated_completions",
    "disable_dropout", "reward_weights", "log_completions", "num_completions_to_print",
    "use_vllm", "vllm_mode", "vllm_gpu_memory_utilization", "vllm_max_model_length",
    "vllm_enable_sleep_mode", "vllm_importance_sampling_correction", "cast_lm_head_to_fp32",
    "shuffle_dataset",
)


# ---- rewards ------------------------------------------------------------------------------

def decision_reward(completions: list[str], reference_decision: list[str],
                    lang: list[str] | None = None, log_metric=None, **_) -> list[float]:
    """DECISION_REWARD when the parsed decision is the attribute-free reference, else 0.

    Also reports the share of sampled completions that hire, per reference side: the
    operating point that DPO's round 1 lost first.
    """
    langs = lang or ["en"] * len(completions)
    decisions = [parse_output(c, g).decision for c, g in zip(completions, langs)]
    references = [str(r).lower() for r in reference_decision]
    if log_metric is not None:
        for side in (HIRE, REJECT):
            mine = [d for d, r in zip(decisions, references) if r == side]
            if mine:
                log_metric(f"sampled_hire_rate_ref_{side}", sum(d == HIRE for d in mine) / len(mine))
    return [DECISION_REWARD if d == r else 0.0 for d, r in zip(decisions, references)]


def well_formed(completion: str) -> bool:
    """Exactly one JSON object, nothing around it, with string `decision` and `feedback` and a
    feedback of FEEDBACK_WORDS words. Stricter than parse_output, which repairs and extracts."""
    try:
        obj = json.loads(completion.strip())
    except (json.JSONDecodeError, AttributeError):
        return False
    if not isinstance(obj, dict) or set(obj) != {"decision", "feedback"}:
        return False
    if not (isinstance(obj["decision"], str) and isinstance(obj["feedback"], str)):
        return False
    low, high = FEEDBACK_WORDS
    return low <= len(obj["feedback"].split()) <= high


def format_reward(completions: list[str], **_) -> list[float]:
    return [FORMAT_REWARD if well_formed(c) else 0.0 for c in completions]


REWARD_FUNCS = [decision_reward, format_reward]


# ---- data ---------------------------------------------------------------------------------

def render_prompts(dataset, tokenizer, chat_template_kwargs: dict | None = None):
    """`prompt` through the audit's chat template, as `_to_conversational` renders a DPO
    prompt; every other column stays for the reward functions."""
    from ..eval.backends import _chat_wrap

    return dataset.map(lambda row: {"prompt": _chat_wrap(tokenizer, row["prompt"],
                                                         chat_template_kwargs)})


# ---- the policy's weights into vLLM ---------------------------------------------------------

def vllm_param_name(name: str, multimodal: bool) -> str:
    """A text-model parameter name as the checkpoint names it, for a multimodal vLLM model."""
    if multimodal and name.startswith("model.") and not name.startswith(
            ("model.language_model.", "model.visual.")):
        return "model.language_model." + name[len("model."):]
    return name


def merged_named_params(model, multimodal: bool = False):
    """(vLLM name, merged weight) for every parameter the served model holds.

    A LoRA layer yields base + B @ A * scaling, computed in fp32 and cast to the base dtype; the
    base weight itself is never written. LoRA's own parameters are skipped, they exist only
    inside the merged weight.
    """
    modules = dict(model.named_modules())
    adapter = getattr(model, "active_adapter", "default")
    prefix = getattr(model, "prefix", "lora_")
    for name, param in model.named_parameters():
        if prefix in name:
            continue
        clean = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
        weight = param.data
        if name.endswith(".base_layer.weight"):
            owner = modules[name[: -len(".base_layer.weight")]]
            if adapter in getattr(owner, "lora_A", {}):
                a = owner.lora_A[adapter].weight.detach().float()
                b = owner.lora_B[adapter].weight.detach().float()
                delta = b @ a
                if getattr(owner, "fan_in_fan_out", False):
                    delta = delta.T
                weight = (weight.float() + delta * owner.scaling[adapter]).to(param.dtype)
        yield vllm_param_name(clean, multimodal), weight


def vllm_model(vllm_generation):
    """The model object inside a colocated TRL engine (the same path TRL's own sync uses)."""
    return vllm_generation.llm.llm_engine.model_executor.driver_worker.model_runner.model


def install_weight_sync(vllm_generation, policy) -> None:
    """Routes TRL's colocated sync through `merged_named_params`."""
    served = type(vllm_model(vllm_generation)).__name__
    multimodal = served.endswith("ForConditionalGeneration")
    log.info("vLLM serves %s; pushing merged weights%s", served,
             " under the checkpoint's model.language_model. names" if multimodal else "")
    vllm_generation._iter_named_params = lambda: merged_named_params(policy, multimodal)


def patch_vllm_engine_kwargs(extra: dict) -> None:
    """Adds `extra` to the `LLM(...)` call TRL makes for its colocated engine."""
    if not extra:
        return
    import functools

    from trl.generation import vllm_generation as VG

    original = getattr(VG.LLM, "__wrapped_by_grpo__", VG.LLM)

    @functools.wraps(original)
    def build(*args, **kwargs):
        return original(*args, **{**kwargs, **extra})

    build.__wrapped_by_grpo__ = original
    VG.LLM = build
    log.info("vLLM engine kwargs added to TRL's LLM(...): %s", extra)


# ---- callbacks ----------------------------------------------------------------------------

def _op_control():
    """scripts/operating_point_control.py as a module: its `validate` checks and thresholds
    are the ones the sync is held to."""
    import importlib.util
    import sys

    from ..utils.config import REPO_ROOT

    name = "operating_point_control"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, REPO_ROOT / "scripts" / "operating_point_control.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _broadcast(value):
    """Rank 0's value on every rank (identity without torch.distributed)."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return value
    box = [value]
    dist.broadcast_object_list(box, src=0)
    return box[0]


def _sync_validation_class():
    from transformers import TrainerCallback

    class SyncValidationCallback(TrainerCallback):
        """Does the vLLM engine serve the policy? Checked on every rank's own engine when a
        segment starts (step 0: base weights replaced by SFT + a zero adapter) and every
        `every` steps after (a trained adapter), against HuggingFace + PEFT on the same rows.

        vLLM margins come from `logprob_token_ids` after the forced decision prefix, the
        op-margin-v2 scorer; HF margins from eval/margins.py. The checks and thresholds are
        operating_point_control's `validate`. A failure stops the run before it trains on
        rollouts from the wrong model.
        """

        def __init__(self, trainer, prompts: list[str], output_dir, every: int,
                     template_kwargs: dict | None, lang: str = "en"):
            self.trainer = trainer
            self.prompts = prompts
            self.output_dir = Path(output_dir)
            self.every = every
            self.template_kwargs = template_kwargs
            self.lang = lang

        def validate(self, step: int) -> dict:
            from vllm import SamplingParams

            from ..eval.backends import _chat_wrap
            from ..eval.margins import PREFIX, WORDS, decision_margins, decision_token_ids

            validation_summary = _op_control().validation_summary
            trainer, tokenizer = self.trainer, self.trainer.processing_class
            trainer.vllm_generation.sync_weights()
            ids = decision_token_ids(tokenizer, WORDS[self.lang])
            texts = [_chat_wrap(tokenizer, p, self.template_kwargs) + PREFIX
                     for p in self.prompts]
            params = SamplingParams(max_tokens=1, temperature=0.0, logprob_token_ids=list(ids))
            outs = trainer.vllm_generation.llm.generate(texts, params, use_tqdm=False)
            lp = [o.outputs[0].logprobs[0] for o in outs]
            vllm_margin = [d[ids[0]].logprob - d[ids[1]].logprob for d in lp]
            decided = sum(o.outputs[0].token_ids[0] in ids for o in outs) / len(outs)
            hf_margin = decision_margins(trainer.accelerator.unwrap_model(trainer.model),
                                         tokenizer, self.prompts, lang=self.lang,
                                         batch_size=4, chat_template_kwargs=self.template_kwargs)
            summary = validation_summary(vllm_margin, hf_margin)
            return {"step": step, "n": len(texts), "vllm_decision_token_share": decided,
                    **summary}

        def _run(self, state, control, when: str):
            result = self.validate(state.global_step)
            passed = result["passed"]
            if state.is_world_process_zero:
                with (self.output_dir / "sync_validation.jsonl").open("a") as fh:
                    fh.write(json.dumps({**result, "when": when}) + "\n")
            log.info("vLLM sync %s at step %d: agreement %.4f, rank correlation %.4f, max |diff| "
                     "%.3f", "PASSED" if passed else "FAILED", state.global_step,
                     result["decision_agreement_hf"], result["rank_correlation_hf"],
                     result["abs_diff_hf"]["max"])
            # Every rank checks its own engine; one bad engine stops all of them.
            import torch
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor([0 if passed else 1], device=self.trainer.accelerator.device)
                dist.all_reduce(flag)
                passed = int(flag.item()) == 0
            if not passed:
                raise RuntimeError(f"vLLM does not serve the policy ({when}, step "
                                   f"{state.global_step}): see sync_validation.jsonl")

        def on_train_begin(self, args, state, control, **kwargs):
            self._run(state, control, "segment start")

        def on_step_end(self, args, state, control, **kwargs):
            if self.every and state.global_step % self.every == 0:
                self._run(state, control, "after step")

    return SyncValidationCallback


def projected_hire_rate(metrics: dict, baseline: dict, anchor: float, ref_share: float) -> dict:
    """The monitor's hire rate on the benchmark's scale.

    The monitor rows are half hire-reference, the benchmark's reference hires `ref_share`, so
    the per-side rates are reweighted to that share. The pool's prompts differ from the
    benchmark's, so the level still differs: the change since SFT is added to SFT's benchmark
    hire rate (`anchor`), and that is what the band is checked against.
    """
    def weighted(m: dict) -> float:
        return ref_share * m["hire_rate_ref_hire"] + (1 - ref_share) * m["hire_rate_ref_reject"]

    now, then = weighted(metrics), weighted(baseline)
    return {"hire_rate_weighted": now, "hire_rate_weighted_sft": then,
            "hire_rate_projected": anchor + now - then}


def _decision_monitor_class():
    from transformers import TrainerCallback

    class DecisionMonitorCallback(TrainerCallback):
        """Greedy decisions on fixed held-out rows every `every` steps (decision_eval's
        generate and metrics, rank 0), logged to W&B and decision_eval.jsonl; stops the run,
        saving first, when the projected hire rate leaves `band`.

        The SFT baseline is the same model with the adapter disabled, scored once per run.
        """

        def __init__(self, rows, tokenizer, output_dir, every: int, band: tuple[float, float],
                     anchor: float, ref_share: float, template_kwargs: dict | None,
                     max_new_tokens: int = 96, lang: str = "en"):
            self.rows = rows.reset_index(drop=True)
            self.tokenizer = tokenizer
            self.output_dir = Path(output_dir)
            self.every = every
            self.band = band
            self.anchor = anchor
            self.ref_share = ref_share
            self.template_kwargs = template_kwargs
            self.max_new_tokens = max_new_tokens
            self.lang = lang
            self.baseline = None
            self.aborted = False

        def _baseline_path(self) -> Path:
            return self.output_dir / "decision_eval_sft.json"

        def _evaluate(self, model) -> tuple[list[str], dict]:
            from .decision_eval import decision_metrics, generate

            outputs = generate(model, self.tokenizer, self.rows["prompt"].tolist(),
                               max_new_tokens=self.max_new_tokens,
                               template_kwargs=self.template_kwargs)
            return outputs, decision_metrics(self.rows, outputs, self.lang)

        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if not state.is_world_process_zero:
                return
            path = self._baseline_path()
            if path.is_file():
                self.baseline = json.loads(path.read_text())
                return
            with model.disable_adapter():
                _, self.baseline = self._evaluate(model)
            path.write_text(json.dumps(self.baseline, indent=2))
            log.info("decision monitor SFT baseline: hire rate %.3f (ref hire %.3f, ref reject "
                     "%.3f), agree %.3f", self.baseline["hire_rate"],
                     self.baseline["hire_rate_ref_hire"], self.baseline["hire_rate_ref_reject"],
                     self.baseline["agree_reference"])

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if not self.every or state.global_step % self.every:
                return
            stop = False
            if state.is_world_process_zero:
                from .decision_eval import record

                outputs, metrics = self._evaluate(model)
                metrics.update(projected_hire_rate(metrics, self.baseline, self.anchor,
                                                   self.ref_share))
                low, high = self.band
                stop = not low <= metrics["hire_rate_projected"] <= high
                metrics["out_of_band"] = stop
                record(self.output_dir, state.global_step, self.rows, outputs, metrics)
                log.info("decision monitor at step %d: agree %.3f, projected hire rate %.3f "
                         "(band %.2f-%.2f), invalid %.3f", state.global_step,
                         metrics["agree_reference"], metrics["hire_rate_projected"], low, high,
                         metrics["invalid_rate"])
            if _broadcast(stop):
                log.error("projected hire rate left the band at step %d: saving and stopping",
                          state.global_step)
                self.aborted = True
                control.should_save = True
                control.should_training_stop = True

    return DecisionMonitorCallback


# ---- training -----------------------------------------------------------------------------

def grpo_config_candidates(cfg: dict) -> dict:
    return {key: cfg[key] for key in GRPO_CONFIG_PASSTHROUGH if key in cfg}


def train(cfg: dict) -> C.TrainArtifacts:
    from ..utils.seed import set_seed

    set_seed(cfg.get("seed", 42))
    C.wandb_init(cfg)
    output_dir = C.resolve_run_dir(cfg)
    if C.is_main_process():
        C.wandb_resume(cfg, output_dir)

    splits = C.load_generated_dataset(cfg, cfg.get("data_kind", "grpo"))
    model, tokenizer = C.load_model_and_tokenizer(cfg)
    chat_kwargs = cfg.get("chat_template_kwargs")
    train_ds = render_prompts(splits["train"], tokenizer, chat_kwargs)

    base_args = C.training_arguments(cfg, output_dir, has_eval=False,
                                     n_train=len(train_ds)).to_dict()
    # No TRL evaluation (the decision monitor is the eval), but checkpoints all the same.
    base_args.update(save_strategy="steps", save_steps=int(cfg.get("save_steps", 25)))
    if not (cfg.get("use_vllm") and cfg.get("vllm_mode") == "colocate"):
        raise ValueError("this trainer expects use_vllm: true and vllm_mode: colocate")
    patch_vllm_engine_kwargs(cfg.get("vllm_engine_kwargs") or {})

    from trl import GRPOConfig, GRPOTrainer

    args = GRPOConfig(**base_args,
                      **C.supported_kwargs(GRPOConfig, grpo_config_candidates(cfg), "GRPOConfig"))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=REWARD_FUNCS,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=C.peft_config(cfg),
        callbacks=C.memory_guard_callbacks(cfg),
    )
    policy = trainer.accelerator.unwrap_model(trainer.model)
    install_weight_sync(trainer.vllm_generation, policy)

    from .decision_eval import eval_rows

    monitor = None
    if "validation" in splits:
        rows = splits["validation"].to_pandas()
        seed = int(cfg.get("seed", 42))
        check = eval_rows(rows, int(cfg.get("sync_validate_rows", 256)), seed)
        trainer.add_callback(_sync_validation_class()(
            trainer, check["prompt"].tolist(), output_dir,
            int(cfg.get("sync_validate_steps", 0)), chat_kwargs))
        band = cfg.get("hire_rate_band", [0.20, 0.40])
        monitor = _decision_monitor_class()(
            eval_rows(rows, int(cfg.get("decision_eval_rows", 256)), seed), tokenizer,
            output_dir, int(cfg.get("decision_eval_steps", 25)), (float(band[0]), float(band[1])),
            float(cfg.get("hire_rate_anchor", 0.28)),
            float(cfg.get("benchmark_reference_hire_share", 0.33)), chat_kwargs,
            int(cfg.get("decision_eval_max_new_tokens", 96)))
        trainer.add_callback(monitor)
    else:
        log.warning("no validation split: the vLLM sync is NOT validated and no hire-rate "
                    "monitor runs")

    manifest = {
        "mitigation_family": "grpo",
        "objective": "grpo",
        "beta": args.beta,
        "num_generations": args.num_generations,
        "rewards": {"decision": DECISION_REWARD, "format": FORMAT_REWARD,
                    "feedback_words": list(FEEDBACK_WORDS)},
        "n_train": len(train_ds),
        "sft_checkpoint": cfg.get("sft_checkpoint"),
    }
    # Before training as well: a run cut by a time limit never reaches the final write, and
    # its checkpoint-N adapters need `sft_checkpoint` to be merged with their SFT parent.
    if C.is_main_process():
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": "training"})
    resume = C.last_checkpoint(output_dir) if cfg.get("resume") else None
    if resume:
        log.info("resuming from %s", resume)
    log.info("GRPO %s on %d prompts x %d generations", cfg["base_model"], len(train_ds),
             args.num_generations)
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    status = "aborted: hire rate left the band" if monitor and monitor.aborted else "done"
    if C.is_main_process():
        tokenizer.save_pretrained(str(output_dir))
        C.write_run_manifest(cfg, output_dir, {**manifest, "status": status})
    return C.TrainArtifacts(output_dir, len(train_ds), 0)


def main() -> None:
    C.configure_cuda_allocator()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.setdefault("stage", "grpo")
    if cfg.get("sft_checkpoint"):
        cfg["base_model"] = C.resolve_sft_checkpoint(cfg["sft_checkpoint"])
    artifacts = train(cfg)
    log.info("GRPO complete -> %s", artifacts.output_dir)


if __name__ == "__main__":
    main()
