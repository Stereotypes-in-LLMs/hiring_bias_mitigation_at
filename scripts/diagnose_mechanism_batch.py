"""Replay selected mechanism baselines in their original batches and as singletons.

Diagnostic selection uses observed errors, so it estimates no population effect.
--plan-only is CPU-only. GPU execution reads frozen weights and writes aggregate,
anonymously labelled measurements. It never submits jobs or changes the environment.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from summarize_mechanism_seeds import SUFFIX, IDENTITY, sha256, require, json_clean, same_identity, load_pure_functions


def choose_pairs(selected, pairs, margins, count):
    require(count >= 4 and count % 2 == 0 and count <= 16, "choose an even 4 to 16 diagnostic pairs")
    index = margins.set_index("prompt_uid").margin_fp32
    joined = selected[IDENTITY].merge(pairs, on=["pair_uid", "candidate_id"], validate="one_to_one")
    joined["full_a"] = joined.uid_a.map(index)
    joined["full_b"] = joined.uid_b.map(index)
    joined["observed_max_error"] = np.maximum(abs(joined.m_a - joined.full_a), abs(joined.m_b - joined.full_b))
    require(np.isfinite(joined.observed_max_error).all(), "missing diagnostic baseline")
    worst = joined.sort_values(["observed_max_error", "pair_uid"], ascending=[False, True]).head(count // 2).copy()
    low = joined[~joined.pair_uid.isin(worst.pair_uid)].sort_values(["observed_max_error", "pair_uid"]).head(count // 2).copy()
    require(len(worst) + len(low) == count, "too few distinct diagnostic pairs")
    worst["pair_label"] = [f"W{i+1:02d}" for i in range(len(worst))]
    low["pair_label"] = [f"L{i+1:02d}" for i in range(len(low))]
    return pd.concat([worst, low], ignore_index=True)


def original_full_batches(part_uids, texts_by_uid, batch_size, wanted):
    """02_margins.run_pass sorts by text CHARACTER length within each stored part."""
    lengths = np.asarray([len(texts_by_uid[u]) for u in part_uids])
    order = np.argsort(lengths, kind="stable")
    ordered = np.asarray(part_uids)[order].tolist()
    return [(start, ordered[start:start + batch_size]) for start in range(0, len(ordered), batch_size)
            if wanted.intersection(ordered[start:start + batch_size])]


def pair_metrics(records):
    frame = pd.DataFrame(records)
    rows = []
    for (context, repeat, label), g in frame.groupby(["context", "repeat", "pair_label"], sort=False):
        require(set(g.side) == {"a", "b"} and len(g) == 2, "incomplete diagnostic pair")
        a, b = g.set_index("side").margin.loc[["a", "b"]]
        rows.append({"context": context, "repeat": repeat, "pair_label": label,
                     "margin_a": a, "margin_b": b, "dm": a - b, "abs_dm": abs(a - b),
                     "crosses_zero": bool((a > 0) != (b > 0)),
                     "crossing_interval_low": min(a, b), "crossing_interval_high": max(a, b)})
    result = pd.DataFrame(rows)
    result["abs_dm_rank_diagnostic_subset"] = result.groupby(["context", "repeat"]).abs_dm.rank(ascending=False, method="min")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="dpo4880_s43", choices=("dpo4880_s43",))
    parser.add_argument("--contrast", default="mil_combat_civ_expl", choices=("mil_combat_civ_expl",))
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/research_followup/mechanism_seeds/batch_diagnostic"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    require(args.repeats == 2, "frozen diagnostic uses two counterordered repetitions")
    root = args.root.resolve()
    inputs = (args.input_root or Path(os.environ.get("HBM_OUTPUT_ROOT", root.parent))).resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    require(not (out / "summary.json").exists(), "completed diagnostic exists; use a new output directory")
    os.environ["HBM_OUTPUT_ROOT"] = str(inputs)
    sys.path.insert(0, str(root / "src"))
    from hiring_bias_mitigation.interp.spec import load_spec_set, resolve_adapter
    from hiring_bias_mitigation.interp import render
    from transformers import AutoTokenizer
    models = load_spec_set(root / "configs/interp/models.yaml")
    spec = models.specs[args.spec]
    base = inputs / "outputs/interp"
    margin_path = base / "margins" / args.spec / "grid-en.parquet"
    meta_path = Path(str(margin_path) + ".meta.json")
    meta = json.loads(meta_path.read_text())
    require(meta["batch_size"] == 8, "diagnostic assumes the recorded full pass batch size is eight")
    cache = base / "patching" / args.spec / "grid" / (args.contrast + SUFFIX)
    cache_identity_path = cache / "poster_followup.meta.json"
    identity = json.loads(cache_identity_path.read_text())
    require(identity["prior_cache_provenance"] == "empty_cache_bound_before_first_compute", "diagnostic requires a fresh bound cache")
    weights = {a: Path(resolve_adapter(models.adapter_paths[a])) / "adapter_model.safetensors" for a in spec.adapters}
    adapter_hashes = {a: sha256(path) for a, path in weights.items()}
    fingerprints = {a: value[:16] for a, value in adapter_hashes.items()}
    require(fingerprints == meta["adapters"] == identity["adapters"], "checkpoint fingerprint differs")
    margins = pd.read_parquet(margin_path)
    require(len(margins) == 31050 and margins.prompt_uid.nunique() == 31050, "full-grid margin coverage differs")
    manifest_path = base / "manifest/contrasts-grid-en.parquet"
    manifest = pd.read_parquet(manifest_path)
    helpers = load_pure_functions(root / "scripts/interp/10_patch.py", ("select_pairs", "load_completed"))
    canonical = helpers["select_pairs"](manifest, args.contrast, 64, .5, 42, "sft_v3")
    selected_path = cache / "selected_pairs.parquet"
    selected = pd.read_parquet(selected_path)
    same_identity(canonical, selected)
    raw = helpers["load_completed"](cache)["pairs"]
    same_identity(selected, raw, ["pair_uid", "candidate_id"])
    diagnostic = choose_pairs(selected, raw, margins, args.pairs)
    wanted = set(diagnostic.uid_a) | set(diagnostic.uid_b)
    require(len(wanted) == args.pairs * 2, "diagnostic pair prompts overlap")
    tokenizer = AutoTokenizer.from_pretrained(models.base_model, cache_dir=str(inputs / "hf-cache/hub"), local_files_only=True)
    tokenizer.padding_side = "left"
    frame = render.render(render.build_frame("grid", meta["data_config"]), tokenizer,
                          meta["data_config"].get("chat_template_kwargs"))
    require(set(frame.prompt_uid) == set(margins.prompt_uid), "current prompt rendering differs")
    texts = frame.set_index("prompt_uid").text.to_dict()
    segment_path = base / "segments/grid-en.parquet"
    segments = pd.read_parquet(segment_path, columns=["prompt_uid", "input_ids", "n_tokens"]).set_index("prompt_uid")
    source_paths = {Path(__file__), root / "scripts/summarize_mechanism_seeds.py", root / "configs/interp/models.yaml",
                    root / "scripts/interp/02_margins.py", root / "scripts/interp/10_patch.py", margin_path,
                    root / "scripts/cscs/poster_mechanism.sbatch", root / "scripts/cscs/env.sh",
                    meta_path, manifest_path, selected_path, cache_identity_path, cache / "run.meta.json", segment_path}
    contexts = {"original_full_batch": [], "original_patch_batch": [], "singleton": []}
    for part in sorted((margin_path.parent / "grid-en").glob("part-*.parquet")):
        uids = pd.read_parquet(part, columns=["prompt_uid"]).prompt_uid.tolist()
        if not wanted.intersection(uids):
            continue
        source_paths.add(part)
        for start, group in original_full_batches(uids, texts, meta["batch_size"], wanted):
            contexts["original_full_batch"].append({"id": f"{part.stem}/{start}", "uids": group})
    for part in sorted(cache.glob("pairs-*.parquet")):
        tag = part.name[len("pairs-"):]
        if not all((cache / f"{kind}-{tag}").exists() for kind in ("real", "atp")):
            continue
        rows = pd.read_parquet(part)
        if not set(rows.pair_uid).intersection(diagnostic.pair_uid):
            continue
        source_paths.add(part)
        rows = rows[["pair_uid"]].merge(selected[IDENTITY], on="pair_uid", validate="one_to_one", sort=False)
        for side in ("a", "b"):
            contexts["original_patch_batch"].append({"id": f"{part.stem}/{side}", "uids": rows[f"uid_{side}"].tolist()})
    aliases = {}
    for row in diagnostic.itertuples():
        for side in ("a", "b"):
            uid = getattr(row, "uid_" + side)
            aliases[uid] = {"pair_label": row.pair_label, "side": side,
                            "original_full_margin": float(getattr(row, "full_" + side)),
                            "original_patch_margin": float(getattr(row, "m_" + side))}
            contexts["singleton"].append({"id": f"{row.pair_label}/{side}", "uids": [uid]})
    context_uids = {u for groups in contexts.values() for group in groups for u in group["uids"]}
    require(context_uids.issubset(segments.index), "batch context lacks cached segment tokens")
    current = tokenizer([texts[u] for u in sorted(context_uids)], add_special_tokens=False)["input_ids"]
    require(all(np.array_equal(ids, segments.loc[u, "input_ids"]) for u, ids in zip(sorted(context_uids), current)),
            "cached tokens differ from fresh tokenization")
    for groups in contexts.values():
        occurrence = [u for g in groups for u in g["uids"] if u in wanted]
        require(len(occurrence) == len(wanted) and set(occurrence) == wanted, "target absent or duplicated in batch replay")
    source_hashes = {str(path): sha256(path) for path in sorted(source_paths)}
    plan = {"spec": args.spec, "contrast": args.contrast, "pairs": args.pairs, "repeats": args.repeats,
            "selection": "Half largest and half smallest observed patch/full baseline discrepancy among the fixed 64 SFT-selected pairs; diagnostic selection, not a population estimate",
            "representation": "Original Qwen Base with SFT-v3 and DPO adapters activated together in HF PEFT; no merge or training",
            "context_order": [list(contexts), list(reversed(contexts))],
            "counts": {name: {"batches_per_repeat": len(groups), "rows_per_repeat": sum(len(g["uids"]) for g in groups)} for name, groups in contexts.items()},
            "total_forwards": args.repeats * sum(len(g) for g in contexts.values()),
            "adapter_sha256": adapter_hashes, "source_sha256": source_hashes,
            "historical_environment": {"torch": meta["torch"], "transformers": meta["transformers"], "kernels": meta["kernels"],
                "fla_version_and_module_path": "not recorded in per-artifact metadata",
                "backend_dispatch_flag": "launcher source configures FLA_DISABLE_BACKEND_DISPATCH=1; per-artifact runtime value not recorded",
                "causal_conv_module_path": "not recorded; availability flag only"},
            "selected_pair_hash": sha256(selected_path),
            "selected_token_hash": __import__("hashlib").sha256(json.dumps([segments.loc[u, "input_ids"].tolist() for u in sorted(wanted)]).encode()).hexdigest(),
            "selected_tokens_match_fresh_render": len(wanted), "context_tokens_match_fresh_render": len(context_uids),
            "batch_context_hashes": {name: [__import__("hashlib").sha256(json.dumps(g["uids"]).encode()).hexdigest() for g in groups] for name, groups in contexts.items()},
            "baseline_replay_scope": "No activation swaps, no source-activation capture; use original baseline readout functions and exact original batch token composition",
            "crossing": "Hire iff margin > 0. Pair interval is [minimum margin, maximum margin), with strict greater-than decision; arbitrary operating-point thresholds require separate calibration"}
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "plan.json"
    if plan_path.exists() and not args.plan_only:
        require(json.loads(plan_path.read_text()) == json_clean(plan), "CPU plan inputs or script changed; regenerate and review plan")
    plan_path.write_text(json.dumps(json_clean(plan), indent=2, allow_nan=False) + "\n")
    diagnostic[["pair_label", "observed_max_error", "m_a", "m_b", "full_a", "full_b"]].to_csv(out / "selection.csv", index=False)
    print(json.dumps({"plan": str(plan_path), "counts": plan["counts"], "total_forwards": plan["total_forwards"]}), flush=True)
    if args.plan_only:
        return
    # GPU imports and model construction begin only after the frozen plan passes.
    import torch
    import transformers
    from hiring_bias_mitigation.interp import spec as SP, margins as M, patching as PT
    from hiring_bias_mitigation.interp.io import kernel_info
    require(kernel_info() == meta["kernels"], "kernel availability differs from original margin pass")
    require(torch.__version__ == meta["torch"] and transformers.__version__ == meta["transformers"], "runtime versions differ")
    require(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", "original FLA dispatch setting required")
    require(os.environ.get("HF_HUB_OFFLINE") == "1", "offline execution required")
    started = time.monotonic()
    model, loaded_tokenizer = SP.load_models(models, [args.spec], device=args.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ro = M.readout(model, loaded_tokenizer, meta["data_config"]["lang"])
    records = []
    with SP.use(model, spec):
        for repeat in range(1, args.repeats + 1):
            order = list(contexts) if repeat == 1 else list(reversed(contexts))
            for context in order:
                for group in contexts[context]:
                    uids = group["uids"]
                    torch.cuda.synchronize()
                    t0 = time.monotonic()
                    if context == "original_full_batch":
                        values = M.score_batch(model, loaded_tokenizer, ro, [texts[u] for u in uids])["margin_fp32"]
                    else:
                        ids, mask, _ = PT._left_pad([segments.loc[u, "input_ids"].tolist() for u in uids], loaded_tokenizer.pad_token_id)
                        values = PT.margins_of(model, ro, ids, mask)
                    torch.cuda.synchronize()
                    elapsed = time.monotonic() - t0
                    for uid, value in zip(uids, values):
                        if uid not in wanted:
                            continue
                        records.append({"context": context, "repeat": repeat, **aliases[uid],
                                        "margin": float(value), "batch_rows": len(uids),
                                        "batch_max_tokens": max(int(segments.loc[u, "n_tokens"]) for u in uids),
                                        "batch_seconds": elapsed})
                    pd.DataFrame(records).to_csv(out / "prompt_margins.csv", index=False)
                print(f"completed repetition {repeat}: {context}", flush=True)
    require(len(records) == args.pairs * 2 * args.repeats * 3, "incomplete diagnostic measurements")
    frame = pd.DataFrame(records)
    original = [{"context": context, "repeat": 0, "pair_label": values["pair_label"], "side": values["side"],
                 "margin": values[field]} for values in aliases.values()
                for context, field in (("recorded_full", "original_full_margin"), ("recorded_patch", "original_patch_margin"))]
    pairs = pair_metrics(records + original)
    pairs.to_csv(out / "pair_metrics.csv", index=False)
    context_summary = {}
    for context, g in frame.groupby("context"):
        wide = g.pivot(index=["pair_label", "side"], columns="repeat", values="margin")
        context_summary[context] = {"repeat_max_abs_difference": float(abs(wide[1] - wide[2]).max()),
            "max_abs_error_vs_recorded_full": float(abs(g.margin - g.original_full_margin).max()),
            "max_abs_error_vs_recorded_patch": float(abs(g.margin - g.original_patch_margin).max())}
    kernel_modules = {}
    for name in ("fla", "fla.ops.gated_delta_rule", "causal_conv1d"):
        try:
            module = importlib.import_module(name)
            kernel_modules[name] = {"file": getattr(module, "__file__", None), "version": getattr(module, "__version__", None)}
        except ImportError:
            kernel_modules[name] = {"available": False}
    distributions = {}
    for name in ("flash-linear-attention", "fla-core", "causal-conv1d"):
        try:
            distributions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
    comparisons = []
    for repeat in (1, 2):
        for other in ("original_patch_batch", "singleton"):
            a = pairs[(pairs.context == "original_full_batch") & (pairs.repeat == repeat)].set_index("pair_label")
            b = pairs[(pairs.context == other) & (pairs.repeat == repeat)].set_index("pair_label").reindex(a.index)
            comparisons.append({"repeat": repeat, "versus_full": other,
                "dm_sign_disagreements": int((np.sign(a.dm) != np.sign(b.dm)).sum()),
                "zero_crossing_disagreements": int((a.crosses_zero != b.crosses_zero).sum()),
                "max_abs_dm_difference": float(abs(a.dm - b.dm).max()),
                "abs_dm_spearman_diagnostic_subset": float(a.abs_dm.rank().corr(b.abs_dm.rank()))})
    summary = {"completed": True, "spec": args.spec, "pairs": args.pairs, "repeats": args.repeats,
        "elapsed_seconds_including_model_load": time.monotonic() - started,
        "device": torch.cuda.get_device_name(), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "context_results": context_summary, "pair_comparisons": comparisons,
        "scope": plan["selection"], "representation": plan["representation"],
        "interpretation_limit": "A completed replay is not a declaration of invariance. Selection is conditioned on observed discrepancies. No numerical difference is automatically classified as harmless noise. These baselines do not rerun intervention effects or establish that their rank order is batch-invariant. Historical artifact metadata lacks exact FLA version/module path and dispatch flags, so failure to reproduce historical values cannot be attributed solely to batching.",
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__, "kernels": kernel_info(),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "fla_disable_backend_dispatch": os.environ.get("FLA_DISABLE_BACKEND_DISPATCH"),
                    "kernel_modules": kernel_modules, "kernel_distributions": distributions,
                    "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "float32_matmul_precision": torch.get_float32_matmul_precision()}, "plan_sha256": sha256(plan_path)}
    (out / "summary.json").write_text(json.dumps(json_clean(summary), indent=2, allow_nan=False) + "\n")
    print(json.dumps(json_clean(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
