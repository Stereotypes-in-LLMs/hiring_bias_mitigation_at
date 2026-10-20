"""Folds a LoRA adapter into its base model, for serving without LoRA.

vLLM's LoRA path does not reproduce Qwen3.5 adapters faithfully: on the same benchmark prompts
the base model's decisions agree 98.9% between HuggingFace and vLLM, an SFT adapter's only
82.5%. The adapter changed 17.5% of decisions under HF + PEFT, the framework it was trained
in, and 0.3% under the vLLM audit. Audits therefore serve adapters as merged checkpoints,
which vLLM loads with no LoRA code involved (`eval/runner.py`).

A DPO adapter fitted on top of an SFT adapter is merged together with it, SFT first: see
`adapter_chain`.

A merge into bf16 also drops the parts of an adapter's update smaller than half a bf16 unit of
the weight they are added to, so the served model can lag the trained one. `merge` measures how
much of each adapter's update survived (`merge_retention`) and records it beside the merged
weights; `measure_retention` does the same for a model merged before the measurement existed.
The evaluation standard (`docs/EVAL_STANDARD.md`) requires the last adapter of the chain, the
arm being evaluated, to keep at least 80% of its update.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from hiring_bias_mitigation.utils.config import resolve_output_path
from hiring_bias_mitigation.utils.logging import get_logger

log = get_logger(__name__)


def merged_dir_for(adapter: str | Path) -> Path:
    """`sft/x` -> `merged/x`; `sft/x/checkpoint-250` -> `merged/x--checkpoint-250`."""
    adapter = Path(adapter)
    name = adapter.name
    if name.startswith("checkpoint-"):
        name = f"{adapter.parent.name}--{name}"
    return Path(resolve_output_path("outputs/merged")) / name


def adapter_chain(adapter: str | Path) -> list[Path]:
    """The adapters that make up a trained model, parents first: [sft, dpo] for a DPO run that
    continued from an SFT run, [adapter] otherwise.

    DPO on top of SFT merges the SFT adapter into the base in memory and fits a fresh LoRA on
    that (`training_common.load_model_and_tokenizer`). PEFT then records the plain base as the
    new adapter's `base_model_name_or_path`, so applying the DPO adapter alone rebuilds base +
    DPO and silently drops SFT. The run's `run_config.json` names the `sft_checkpoint` it
    started from; a `checkpoint-N` directory reads its run's file.
    """
    chain = [Path(adapter)]
    while (parent := _parent_adapter(chain[0])) is not None:
        if parent in chain:
            raise ValueError(f"adapter chain loops back to {parent}: {chain}")
        chain.insert(0, parent)
    return chain


def _parent_adapter(adapter: Path) -> Path | None:
    run_dir = adapter.parent if adapter.name.startswith("checkpoint-") else adapter
    manifest = run_dir / "run_config.json"
    if not manifest.is_file():
        return None
    spec = json.loads(manifest.read_text()).get("sft_checkpoint")
    if not spec:
        return None
    from .training_common import resolve_sft_checkpoint

    return Path(resolve_sft_checkpoint(spec))


def merge_chain(model, chain: list[Path], retention: list | None = None):
    """Applies and merges each adapter of `chain` in order, parents first.

    With a `retention` list, each adapter's merge is measured layer by layer and one record per
    adapter is appended to it (see `merge_measured`). The merged weights are the same either way.
    """
    import warnings

    from peft import PeftModel

    for adapter in chain:
        peft_model = PeftModel.from_pretrained(model, str(adapter))
        if retention is not None:
            retention.append({"adapter": str(adapter), **merge_measured(peft_model)})
        with warnings.catch_warnings():
            # After merge_measured every measured layer is merged already; PEFT says so.
            warnings.filterwarnings("ignore", message="All adapters are already merged")
            model = peft_model.merge_and_unload()
    return model


def merge_measured(peft_model) -> dict:
    """Merges every LoRA linear layer in place and measures what the merge kept.

    The exact update of a layer is `B @ A * scaling` in float32; the merge writes `W' - W`
    into the base dtype. `kept_fraction` is the written change projected on the exact update,
    sum((W' - W) * D) / sum(D * D): 1.0 when all of it survived, near 0 when rounding dropped
    it. `unchanged` is the share of weights the merge left bit-identical. Layers PEFT merges
    without a `lora_A` linear (embeddings) are merged by `merge_and_unload` afterwards and are
    not measured; `layers` counts the measured ones.
    """
    import torch
    from peft.tuners.lora import LoraLayer

    kept = total = unchanged = count = 0.0
    layers = 0
    with torch.no_grad():
        for module in peft_model.modules():
            if not isinstance(module, LoraLayer):
                continue
            names = [n for n in module.active_adapters
                     if n in module.lora_A and n not in module.merged_adapters]
            if not names:
                continue
            weight = module.get_base_layer().weight
            before = weight.detach().float().clone()
            exact = torch.zeros_like(before)
            for name in names:
                delta = module.lora_B[name].weight.float() @ module.lora_A[name].weight.float()
                if getattr(module, "fan_in_fan_out", False):
                    delta = delta.T
                exact += delta * module.scaling[name]
            module.merge(adapter_names=names)
            change = weight.detach().float() - before
            kept += float((change * exact).sum())
            total += float((exact * exact).sum())
            unchanged += float((change == 0).sum())
            count += change.numel()
            layers += 1
    return {"kept_fraction": kept / total if total else 1.0,
            "unchanged": unchanged / count if count else 1.0,
            "layers": layers}


def conform_to_base(out: Path, base_name: str, base_dir: Path) -> None:
    """Give the merged checkpoint the base's exact layout.

    Training loads the text model only (`Qwen3_5ForCausalLM`), so the save lacks the base's
    vision tower and carries a text-only config. vLLM would then serve it through a different
    model class than the baseline audit. Copying the base config and the untouched tensors the
    save lacks makes the merged checkpoint differ from the base in the merged weights only.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    saved_arch = json.loads((out / "config.json").read_text()).get("architectures")
    base_arch = json.loads((base_dir / "config.json").read_text()).get("architectures")
    if saved_arch == base_arch:
        # The save already is the base's model class (e.g. Gemma-3 based LAPA, which loads as
        # Gemma3ForConditionalGeneration). Its tensor names may follow the newer transformers
        # layout rather than the base checkpoint's, which vLLM reads either way.
        log.info("merged save already has the base architecture %s; kept as saved", saved_arch)
        return

    base_map = json.loads((base_dir / "model.safetensors.index.json").read_text())["weight_map"]
    ours: dict[str, str] = {}
    for f in sorted(out.glob("*.safetensors")):
        if f.is_symlink():
            raise SystemExit(f"{f} is a link into the HF cache, not a merged tensor file")
        with safe_open(f, "pt") as fh:
            ours.update(dict.fromkeys(fh.keys(), f.name))
    unknown = set(ours) - set(base_map)
    if unknown:
        raise SystemExit(f"merged tensors the base does not have: {sorted(unknown)[:5]}")
    missing = sorted(set(base_map) - set(ours))
    if missing:
        from huggingface_hub import snapshot_download

        shards = sorted({base_map[k] for k in missing})
        base_w = Path(snapshot_download(base_name, allow_patterns=shards))
        extra = {}
        for shard in shards:
            with safe_open(base_w / shard, "pt") as fh:
                extra.update({k: fh.get_tensor(k) for k in missing if base_map[k] == shard})
        save_file(extra, out / "base-untouched.safetensors", metadata={"format": "pt"})
        ours.update(dict.fromkeys(extra, "base-untouched.safetensors"))
    total = sum((out / f).stat().st_size for f in set(ours.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(ours.items()))}, indent=2))
    shutil.copy2(base_dir / "config.json", out / "config.json")
    log.info("conformed to base layout: %d merged + %d base tensors", len(ours) - len(missing),
             len(missing))


def merge(adapter: str | Path, force: bool = False) -> Path:
    """Merges `adapter` into its base once and returns the merged directory."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(resolve_output_path(str(adapter)))
    chain = adapter_chain(adapter)
    mtime = _chain_mtime(chain)
    out = merged_dir_for(adapter)
    marker = out / "merged_from.json"
    if marker.exists() and not force:
        recorded = json.loads(marker.read_text())
        # Retraining writes a new adapter to the same path; a stale merge must not be reused.
        if recorded.get("adapter_mtime") == mtime:
            log.info("%s already merged", out)
            return out
        log.info("adapter %s changed since it was merged; re-merging", adapter)

    bases = [json.loads((a / "adapter_config.json").read_text())["base_model_name_or_path"]
             for a in chain]
    base_name = bases[0]
    if len(set(bases)) > 1:
        log.warning("adapters record different bases %s; loading %s", bases, base_name)
    log.info("merging %s into %s", " then ".join(map(str, chain)), base_name)
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16,
                                                 device_map="cpu", trust_remote_code=True)
    retention: list[dict] = []
    model = merge_chain(model, chain, retention)
    for record in retention:
        log.info("merge kept %.3f of %s (%d layers)", record["kept_fraction"], record["adapter"],
                 record["layers"])
    if out.exists():
        shutil.rmtree(out)  # never mix shards from an earlier, interrupted merge
    out.mkdir(parents=True)
    model.save_pretrained(out, safe_serialization=True)
    del model
    AutoTokenizer.from_pretrained(base_name, trust_remote_code=True).save_pretrained(out)
    # Keep whatever else the base ships for loading (chat template, generation config,
    # preprocessor files a multimodal checkpoint declares). The snapshot dir also holds any
    # base weights downloaded earlier; copy only the small loading files, never tensors or the
    # base's weight index.
    base_dir = Path(snapshot_download(base_name, allow_patterns=["*.json", "*.jinja", "*.txt"]))
    for f in base_dir.iterdir():
        if (f.suffix in {".json", ".jinja", ".txt"} and f.name != "model.safetensors.index.json"
                and not (out / f.name).exists()):
            shutil.copy2(f, out / f.name)
    conform_to_base(out, base_name, base_dir)
    marker.write_text(json.dumps({"base_model": base_name, "adapter": str(adapter),
                                  "adapters": [str(a) for a in chain],
                                  "adapter_mtime": mtime,
                                  "merge_retention": retention}, indent=2))
    log.info("wrote %s", out)
    return out


def measure_retention(adapter: str | Path) -> Path:
    """Measures the merge of an already merged adapter without touching its weights.

    Rebuilds the merge in memory exactly as `merge` does (bf16 base on CPU, the chain parents
    first) and writes `merge_retention.json` into the merged directory. For models merged before
    `merge` recorded the measurement.
    """
    import torch
    from transformers import AutoModelForCausalLM

    adapter = Path(resolve_output_path(str(adapter)))
    chain = adapter_chain(adapter)
    out = merged_dir_for(adapter)
    base_name = json.loads((chain[0] / "adapter_config.json").read_text())[
        "base_model_name_or_path"]
    model = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.bfloat16,
                                                 device_map="cpu", trust_remote_code=True)
    retention: list[dict] = []
    merge_chain(model, chain, retention)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "merge_retention.json"
    path.write_text(json.dumps({"base_model": base_name, "adapters": [str(a) for a in chain],
                                "adapter_mtime": _chain_mtime(chain),
                                "merge_retention": retention}, indent=2))
    log.info("wrote %s", path)
    return path


def recorded_retention(merged: str | Path) -> list[dict] | None:
    """The merge measurement recorded for a merged directory, or None when there is none."""
    merged = Path(merged)
    for name in ("merged_from.json", "merge_retention.json"):
        path = merged / name
        if path.is_file():
            records = json.loads(path.read_text()).get("merge_retention")
            if records:
                return records
    return None


def _adapter_mtime(adapter: Path) -> float:
    return max(f.stat().st_mtime for f in adapter.glob("adapter_model*"))


def _chain_mtime(chain: list[Path]) -> float | list[float]:
    # A single adapter keeps the plain number, so merges recorded before chains stay valid.
    mtimes = [_adapter_mtime(a) for a in chain]
    return mtimes[0] if len(mtimes) == 1 else mtimes
