"""Inference-only comparison of a final PEFT adapter and its served bf16 merge.

Reconstruct the common SFT checkpoint once, in the same CPU bf16 merge order as
the serving pipeline. Only the final distillation adapter remains unmerged in
the PEFT condition. Do not stack an unmerged SFT adapter with the final adapter.
All conditions use identical HF inputs, batches, SDPA and fp32/native readouts.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ARMS = {
    "full_r32": "decision_distill_seed{seed}",
    "window_r32": "decision_distill_l15_20_seed{seed}",
    "window_r163": "decision_distill_l15_20_r163_seed{seed}",
}
KEYS = ["dataset", "pair_id", "protected_group", "protected_attr", "condition",
        "candidate_id", "job_id", "group_id", "prompt_uid"]
READOUTS = ["margin_fp32", "margin_bf16", "logprob_hire", "logprob_reject", "top1"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def freeze(path: Path, value) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing stale resume: {path} differs from current inputs")
    else:
        write_json(path, value)


def files_identity(path: Path, weights: bool = True) -> dict:
    """Content identity; no timestamps or job IDs enter the resume signature."""
    names = {p for pat in ("*.json", "*.jinja", "tokenizer.*") for p in path.glob(pat)}
    if weights:
        names.update(path.glob("*.safetensors"))
    return {"path": str(path.resolve()), "files": {
        p.name: sha256(p) for p in sorted(names) if p.is_file()}}


def validate_chain(chain, expected_sft: Path, adapter: Path, marker: dict) -> None:
    resolved = [str(Path(p).resolve()) for p in chain]
    expected = [str(expected_sft.resolve()), str(adapter.resolve())]
    if resolved != expected:
        raise ValueError(f"Expected exactly common SFT then final adapter: {resolved}")
    recorded = [str(Path(p).resolve()) for p in marker.get("adapters", [])]
    if recorded != expected:
        raise ValueError(f"Served checkpoint has a different adapter chain: {recorded}")


def select_pilot(frame, n_pairs: int):
    """Length-spaced complete pairs, chosen without looking at model outputs."""
    import numpy as np

    if n_pairs < 1:
        raise ValueError("pilot needs at least one pair")
    sizes = frame.assign(_length=frame.text.str.len()).groupby("pair_id")["_length"].median()
    order = sorted(sizes.index, key=lambda k: (sizes[k], str(k)))
    indices = np.linspace(0, len(order) - 1, min(n_pairs, len(order)), dtype=int)
    chosen = {order[i] for i in indices}
    return frame[frame.pair_id.isin(chosen)].reset_index(drop=True)


def summary(a, b, column: str) -> dict:
    import numpy as np

    if a.prompt_uid.tolist() != b.prompt_uid.tolist():
        raise ValueError("score rows differ")
    x, y = a[column].to_numpy(), b[column].to_numpy()
    delta = y - x
    return {"rows": len(x), "mean_signed_b_minus_a": float(delta.mean()),
            "mean_abs": float(np.abs(delta).mean()), "max_abs": float(np.abs(delta).max()),
            "p99_abs": float(np.quantile(np.abs(delta), .99)),
            "decision_agreement_pct": float(100 * ((x > 0) == (y > 0)).mean()),
            "hire_rate_a_pct": float(100 * (x > 0).mean()),
            "hire_rate_b_pct": float(100 * (y > 0).mean())}


def score_artifacts(destination: Path, conditions: list[str], expected_uids: list[str]) -> dict:
    """Validate every output part even when a completed summary already exists."""
    import numpy as np
    import pandas as pd

    records = {}
    for condition in conditions:
        observed = []
        parts = sorted((destination / condition).glob("part-*.parquet"))
        for part in parts:
            frame = pd.read_parquet(part)
            if not set(READOUTS).issubset(frame) or not np.isfinite(frame[READOUTS].to_numpy()).all():
                raise ValueError(f"Invalid completed scores: {part}")
            observed.extend(frame.prompt_uid.tolist())
            records[str(part.relative_to(destination))] = {"sha256": sha256(part), "rows": len(frame)}
        if observed != expected_uids:
            raise ValueError(f"Incomplete completed scores: {condition}")
    return records


def verify_saved_weights(model, saved: Path) -> dict:
    """Bitwise comparison of every loaded unique parameter with served safetensors."""
    import torch
    from safetensors import safe_open

    index = saved / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text())["weight_map"]
    else:
        mapping = {}
        for file in sorted(saved.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as handle:
                mapping.update({key: file.name for key in handle.keys()})
    config_path = saved / "config.json"
    saved_arch = json.loads(config_path.read_text()).get("architectures", []) if config_path.exists() else []
    grouped = {}
    name_map = {}
    for name, parameter in model.named_parameters():
        saved_name = name
        # HF loads the text-only class from the conformed multimodal checkpoint.
        # Its documented text subtree drops the `language_model` component.
        if (saved_name not in mapping and name.startswith("model.")
                and "Qwen3_5ForConditionalGeneration" in saved_arch):
            saved_name = "model.language_model." + name[len("model."):]
        if saved_name not in mapping:
            raise ValueError(f"Served checkpoint is missing parameter {name}")
        name_map[name] = saved_name
        grouped.setdefault(mapping[saved_name], []).append((name, saved_name, parameter))
    if len(set(name_map.values())) != len(name_map):
        raise ValueError("Saved parameter mapping is not one-to-one")
    checked = 0
    for file, values in grouped.items():
        with safe_open(saved / file, framework="pt", device="cpu") as handle:
            for name, saved_name, parameter in values:
                reference = handle.get_tensor(saved_name)
                if parameter.dtype != reference.dtype or not torch.equal(parameter.detach().cpu(), reference):
                    raise ValueError(f"Rebuilt merge differs from served parameter {name}")
                checked += parameter.numel()
    return {"all_unique_parameters_bit_identical": True, "parameter_elements": checked,
            "loaded_parameters": sum(map(len, grouped.values())),
            "name_map_sha256": digest(name_map),
            "qwen_text_prefix_mappings": sum(k != v for k, v in name_map.items())}


def load_base(path: str, device: str = "cpu"):
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        attn_implementation="sdpa").eval()


def score(model, tokenizer, frame, out: Path, batch_size: int, chunk: int):
    import numpy as np
    import pandas as pd
    from hiring_bias_mitigation.interp import margins as M

    out.mkdir(parents=True, exist_ok=True)
    ro = M.readout(model, tokenizer, "en")
    parts = []
    for start in range(0, len(frame), chunk):
        rows = frame.iloc[start:start + chunk]
        path = out / f"part-{start // chunk:05d}.parquet"
        if path.exists():
            part = pd.read_parquet(path)
            if part.prompt_uid.tolist() != rows.prompt_uid.tolist() or not set(READOUTS).issubset(part):
                raise ValueError(f"Invalid cached part {path}")
            if not np.isfinite(part[READOUTS].to_numpy()).all():
                raise ValueError(f"Nonfinite cached scores in {path}")
            parts.append(part)
            continue
        order = np.argsort(rows.text.str.len().to_numpy(), kind="stable")
        texts = rows.text.to_numpy()[order].tolist()
        values = {key: [] for key in READOUTS}
        begun = time.monotonic()
        for offset in range(0, len(texts), batch_size):
            result = M.score_batch(model, tokenizer, ro, texts[offset:offset + batch_size])
            for key in READOUTS:
                values[key].append(result[key])
        part = rows[KEYS].reset_index(drop=True).copy()
        for key in READOUTS:
            part[key] = np.concatenate(values[key])[np.argsort(order)]
        if not np.isfinite(part[READOUTS].to_numpy()).all():
            raise ValueError(f"Nonfinite scores in {path}")
        temp = path.with_suffix(".parquet.tmp")
        part.to_parquet(temp, index=False)
        temp.replace(path)
        print(f"SCORE {out.name} rows {start}:{start + len(rows)} {time.monotonic() - begun:.1f}s", flush=True)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--pilot-pairs", type=int, default=12)
    parser.add_argument("--out-root", type=Path, default=REPO / "reports/research_followup/merge_precision")
    parser.add_argument("--preflight", action="store_true", help="CPU metadata only; never load a model")
    args = parser.parse_args()
    if args.batch_size < 1 or args.chunk < 1 or args.chunk % args.batch_size:
        parser.error("chunk must be a positive multiple of batch size")
    from hiring_bias_mitigation.mitigation import merge as MG
    from hiring_bias_mitigation.interp.io import data_config
    from hiring_bias_mitigation.utils.config import resolve_output_path

    adapter = Path(resolve_output_path("outputs/distill/" + ARMS[args.arm].format(seed=args.seed)))
    sft = Path(resolve_output_path("outputs/sft/qwen3.5-9b_en_only_sft_v3_additional"))
    served = MG.merged_dir_for(adapter)
    served_sft = MG.merged_dir_for(sft)
    chain = MG.adapter_chain(adapter)
    marker = json.loads((served / "merged_from.json").read_text())
    validate_chain(chain, sft, adapter, marker)
    terminal = json.loads((adapter / "checkpoint-341/trainer_state.json").read_text())
    if terminal["global_step"] != 341 or abs(float(terminal["epoch"]) - 1) > .01:
        raise ValueError("Final checkpoint is not the declared completed one-epoch run")
    if sha256(adapter / "adapter_model.safetensors") != sha256(adapter / "checkpoint-341/adapter_model.safetensors"):
        raise ValueError("Saved final adapter differs from declared checkpoint-341")
    base = json.loads((sft / "adapter_config.json").read_text())["base_model_name_or_path"]
    cfg = data_config()
    if args.preflight:
        # Avoid reading tens of GB of model shards on a login node. Full hashes,
        # model loading and parameter comparisons occur only inside the job.
        for checkpoint in (served_sft, served):
            if not list(checkpoint.glob("*.safetensors")):
                raise ValueError(f"Missing served weights: {checkpoint}")
        print(json.dumps({"status": "preflight_passed", "arm": args.arm, "seed": args.seed,
                          "chain": [str(p) for p in chain], "checkpoint": 341,
                          "last_adapter_sha256": sha256(adapter / "adapter_model.safetensors"),
                          "served": str(served), "mode": args.mode}, indent=2))
        return
    source_files = [Path(__file__), REPO / "src/hiring_bias_mitigation/mitigation/merge.py",
                    REPO / "src/hiring_bias_mitigation/interp/margins.py",
                    REPO / "src/hiring_bias_mitigation/interp/render.py"]
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "peft", "safetensors", "numpy", "pandas")}
    common = {"schema": 1, "base": base, "sft": files_identity(sft),
              "served_sft": files_identity(served_sft), "data_config": cfg,
              "batch_size": args.batch_size, "chunk": args.chunk,
              "versions": versions, "code": {str(p.relative_to(REPO)): sha256(p) for p in source_files},
              "base_dtype": "bfloat16", "adapter_autocast_dtype": True,
              "attention": "sdpa", "device_type": "cuda", "readouts": READOUTS,
              "padding_side": "left", "use_cache": False, "tf32": False}
    identity = {"common": common, "arm": args.arm, "seed": args.seed,
                "adapter": files_identity(adapter), "served": files_identity(served),
                "chain": [str(p.resolve()) for p in chain],
                "mode": args.mode, "pilot_pairs": args.pilot_pairs}
    destination = args.out_root / args.mode / f"{args.arm}_seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / ".lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze(destination / "identity.json", identity)
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from hiring_bias_mitigation.interp import render

    if not torch.cuda.is_available():
        raise RuntimeError("GPU inference must run inside a GPU allocation")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    frame = render.render(render.build_frame("grid", cfg), tokenizer, cfg.get("chat_template_kwargs"))
    if len(frame) != 31050 or frame.prompt_uid.duplicated().any():
        raise ValueError("Expected 31,050 distinct benchmark prompts")
    full_input = digest(frame[KEYS + ["text"]].to_dict("records"))
    if args.mode == "pilot":
        frame = select_pilot(frame, args.pilot_pairs)
    inputs = {"full_grid_sha256": full_input, "rows": len(frame),
              "selected_sha256": digest(frame[KEYS + ["text"]].to_dict("records")),
              "prompt_uids": frame.prompt_uid.tolist(),
              "tokenization_sha256": digest([tokenizer(t, add_special_tokens=False)["input_ids"] for t in frame.text])}
    freeze(destination / "inputs.json", inputs)
    if args.mode == "full":
        pilot_path = args.out_root / "pilot" / f"{args.arm}_seed42"
        pilot = json.loads((pilot_path / "summary.json").read_text())
        old = json.loads((pilot_path / "identity.json").read_text())
        old_inputs = json.loads((pilot_path / "inputs.json").read_text())
        if not pilot.get("pilot_passed") or old["common"] != common or old_inputs["full_grid_sha256"] != full_input:
            raise ValueError("Full pass requires a passing same-arm seed42 pilot with identical common inputs")
    if (destination / "summary.json").exists():
        old = json.loads((destination / "summary.json").read_text())
        conditions = ["unmerged", "merged_bf16"] + (["saved_served"] if args.mode == "pilot" else [])
        if score_artifacts(destination, conditions, frame.prompt_uid.tolist()) != old["score_artifacts"]:
            raise ValueError("Completed score artifacts changed; refusing to skip")
        print(f"COMPLETE validated existing result {destination}")
        return
    write_json(destination / "execution.json", {"job": os.environ.get("SLURM_JOB_ID"),
               "device": args.device, "gpu": torch.cuda.get_device_name(args.device),
               "gpu_total_bytes": torch.cuda.get_device_properties(args.device).total_memory,
               "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "started_unix": time.time(), "identity_sha256": digest(identity)})
    begun = time.monotonic()
    # Rebuild on CPU exactly as the saved serving merge; SFT is merged only once.
    model = MG.merge_chain(load_base(base), [sft])
    common_parity = verify_saved_weights(model, served_sft)
    model = PeftModel.from_pretrained(model, str(adapter), autocast_adapter_dtype=True).eval()
    dtypes = sorted({str(p.dtype) for n, p in model.named_parameters() if "lora_" in n})
    model.to(args.device)
    unmerged = score(model, tokenizer, frame, destination / "unmerged", args.batch_size, args.chunk)
    # The saved merge was made on CPU. Do not substitute a CUDA matmul merge.
    model.cpu()
    torch.cuda.empty_cache()
    retention = MG.merge_measured(model)
    model = model.merge_and_unload().eval()
    merged_parity = verify_saved_weights(model, served)
    model.to(args.device)
    merged = score(model, tokenizer, frame, destination / "merged_bf16", args.batch_size, args.chunk)
    results = {"status": "complete", "mode": args.mode, "arm": args.arm, "seed": args.seed,
               "rows": len(frame), "common_sft_weight_parity": common_parity,
               "served_weight_parity": merged_parity, "last_adapter_merge_projection": retention,
               "unmerged_adapter_dtypes": dtypes,
               "scope": "Same HF backend; common SFT already merged; only final adapter changes serving representation.",
               "unmerged_vs_merged": {key: summary(unmerged, merged, key)
                                      for key in ("margin_fp32", "margin_bf16")}}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if args.mode == "pilot":
        model = load_base(str(served), args.device)
        saved = score(model, tokenizer, frame, destination / "saved_served", args.batch_size, args.chunk)
        parity = {key: summary(merged, saved, key) for key in ("margin_fp32", "margin_bf16")}
        results["rebuilt_vs_saved_scores"] = parity
        results["pilot_passed"] = all(v["max_abs"] <= 1e-5 for v in parity.values())
        if not results["pilot_passed"]:
            write_json(destination / "pilot_failure.json", results)
            raise ValueError("Saved merge score reproduction failed; full pass must not run")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    results["wall_seconds"] = time.monotonic() - begun
    results["torch_peak_allocated_gib"] = torch.cuda.max_memory_allocated(args.device) / 2**30
    conditions = ["unmerged", "merged_bf16"] + (["saved_served"] if args.mode == "pilot" else [])
    results["score_artifacts"] = score_artifacts(destination, conditions, frame.prompt_uid.tolist())
    write_json(destination / "summary.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
