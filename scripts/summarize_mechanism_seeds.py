"""CPU-only audit of the nine frozen post-SFT mechanism checkpoints.

Reads existing GPU outputs; never changes inputs. CSVs contain aggregate statistics,
not CV text or candidate identifiers. Fail closed on selection, identity or summary
inconsistencies. Historical cache identity gaps remain explicit warnings.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CONTRASTS = ("mil_combat_civ_impl", "mil_combat_civ_expl",
             "rel_atheist_christian_impl", "gen_male_female_impl")
ARMS = {
    "dpo4880": ("DPO", 42), "dpo4880_s43": ("DPO", 43), "dpo4880_s44": ("DPO", 44),
    "grpo_r4": ("GRPO + lr decay", 42), "grpo_r4_s43": ("GRPO + lr decay", 43),
    "grpo_r4_s44": ("GRPO + lr decay", 44), "distill42": ("Self-distillation", 42),
    "distill43": ("Self-distillation", 43), "distill44": ("Self-distillation", 44),
}
NEW_SPECS = {"dpo4880_s43", "dpo4880_s44", "grpo_r4_s43", "grpo_r4_s44", "distill42", "distill44"}
SPECS = ("sft_v3", *ARMS)
SUFFIX = "-n64-s42-sel_sft_v3"
IDENTITY = ["pair_uid", "candidate_id", "uid_a", "uid_b"]
TARGETS = ((15, 11), (23, 1))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def semantic_hash(frame, columns):
    require(not frame[columns].isna().any().any(), "missing identity values")
    ordered = frame[columns].astype(str).sort_values(columns).reset_index(drop=True)
    return hashlib.sha256(ordered.to_json(orient="values").encode()).hexdigest()


def same_identity(expected, actual, columns=IDENTITY):
    for name, frame in (("expected", expected), ("actual", actual)):
        require(not frame["pair_uid"].duplicated().any(), f"duplicate {name} pair_uid")
    require(len(actual) == len(expected), "selection size differs")
    require(semantic_hash(expected, columns) == semantic_hash(actual, columns),
            "selection identities differ")


def load_pure_functions(path, names):
    """Use the repo's selection/batch rules without importing model/GPU modules."""
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), "canonical helper missing")
    namespace = {"pd": pd, "Path": Path}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def effect_summary(pairs, effects, keys, *, atp=False):
    """Recompute point estimates, preserving the sign convention in patching.summarise.

    The source CSV ci_low/high are CIs for SHARE REMOVED, not logits per pair.
    We validate point estimates but do not relabel or fabricate logit CIs.
    """
    require(not pairs["pair_uid"].duplicated().any(), "duplicate raw pair")
    require(not effects.duplicated(["pair_uid", *keys]).any(), "duplicate raw intervention")
    joined = effects.merge(pairs[["pair_uid", "m_a", "m_b"]], on="pair_uid",
                           validate="many_to_one", how="left", indicator=True)
    require((joined["_merge"] == "both").all(), "intervention on unknown pair")
    dm = joined.m_a - joined.m_b
    change = joined.atp_delta if atp else joined.m_patched - joined.m_a
    joined = joined.assign(removed=-np.sign(dm) * change, size=np.abs(dm))
    require(np.isfinite(joined[["removed", "size"]]).all().all(), "nonfinite raw effect")
    out = joined.groupby(keys, dropna=False).agg(removed=("removed", "sum"),
           size=("size", "sum"), pairs=("pair_uid", "size"),
           removed_logits_per_pair=("removed", "mean")).reset_index()
    require((out.pairs == len(pairs)).all(), "incomplete component/head interventions")
    out["share_removed"] = out.removed / out["size"]
    return out[keys + ["pairs", "removed_logits_per_pair", "share_removed"]]


def compare_summary(raw_summary, recorded, keys):
    require(not recorded.duplicated(keys).any(), "duplicate recorded component/head")
    joined = raw_summary.merge(recorded, on=keys, suffixes=("_raw", "_csv"),
                               validate="one_to_one", how="outer", indicator=True)
    require((joined._merge == "both").all(), "raw and CSV component/head keys differ")
    errors = {}
    for metric in ("pairs", "removed_logits_per_pair", "share_removed"):
        a, b = joined[metric + "_raw"], joined[metric + "_csv"]
        require(np.allclose(a, b, rtol=2e-6, atol=2e-7, equal_nan=False),
                f"raw and CSV disagree on {metric}")
        errors[metric] = float(np.max(np.abs(a - b)))
    return errors


def component_profile(reference, arm):
    keys = ["kind", "layer", "cls"]
    frames = []
    for frame in (reference, arm):
        picked = frame[frame.kind.isin(("mixer", "mlp")) & frame.cls.eq("suffix")]
        require(not picked.duplicated(keys).any(), "duplicate real component")
        frames.append(picked)
    joined = frames[0].merge(frames[1], on=keys, suffixes=("_ref", "_arm"),
                             validate="one_to_one", how="outer", indicator=True)
    require((joined._merge == "both").all(), "real component coverage differs")
    require(len(joined) == 64, "expected 32 mixers and 32 MLPs")
    x, y = joined.removed_logits_per_pair_ref, joined.removed_logits_per_pair_arm
    require(np.isfinite(x).all() and np.isfinite(y).all(), "nonfinite component effect")
    require(np.std(x) > 0 and np.std(y) > 0, "constant component profile")
    return {"real_component_pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "components": len(joined), "positions": "suffix", "source": "real.csv"}


def head_tables(verify, atp):
    real = verify[verify.cls.eq("suffix")].copy()
    approx = atp[atp.kind.eq("head_in") & atp.cls.eq("suffix")].copy()
    keys = ["layer", "head"]
    require(not real.duplicated(keys).any(), "duplicate verified head")
    require(not approx.duplicated(keys).any(), "duplicate AtP head")
    real["real_rank_tested"] = real.removed_logits_per_pair.rank(ascending=False, method="min")
    approx["atp_rank_all"] = approx.removed_logits_per_pair.rank(ascending=False, method="min")
    real["is_control"] = real.control.astype(str).str.lower().isin(("true", "1"))
    rows = []
    for layer, head in TARGETS:
        r = real[(real.layer == layer) & (real["head"] == head)]
        a = approx[(approx.layer == layer) & (approx["head"] == head)]
        require(len(a) == 1, "target absent from AtP inventory")
        row = {"target": f"L{layer}.H{head}", "positions": "suffix", "verified": bool(len(r)),
               "tested_heads": len(real), "atp_heads": len(approx),
               "rank_scope": "descending signed effect among this run's tested heads only",
               "atp_rank_all": int(a.iloc[0].atp_rank_all),
               "atp_removed_logits_per_pair": float(a.iloc[0].removed_logits_per_pair),
               "real_rank_tested": None, "real_removed_logits_per_pair": None,
               "real_share_removed": None, "real_share_ci_low": None,
               "real_share_ci_high": None, "is_control": None}
        if len(r):
            hit = r.iloc[0]
            row.update(real_rank_tested=int(hit.real_rank_tested),
                real_removed_logits_per_pair=float(hit.removed_logits_per_pair),
                real_share_removed=float(hit.share_removed),
                real_share_ci_low=float(hit.ci_low), real_share_ci_high=float(hit.ci_high),
                is_control=bool(hit.is_control))
        rows.append(row)
    top = real.sort_values(["removed_logits_per_pair", "layer", "head"], ascending=[False, True, True]).iloc[0]
    joined = real.merge(approx, on=keys, suffixes=("_real", "_atp"), validate="one_to_one")
    require(len(joined) == len(real), "verified head absent from AtP")
    # Spearman is descriptive on the adaptively chosen tested subset, never a causal route.
    spearman = joined.removed_logits_per_pair_real.rank().corr(joined.removed_logits_per_pair_atp.rank())
    overview = {"tested_heads": len(real), "control_heads": int(real.is_control.sum()),
                "atp_heads": len(approx), "top_real_tested_head": f"L{int(top.layer)}.H{int(top['head'])}",
                "top_real_removed_logits_per_pair": float(top.removed_logits_per_pair),
                "atp_real_spearman_tested_subset": float(spearman)}
    return rows, overview


def full_grid_effects(manifest, reference, arm):
    rows = manifest[manifest.contrast_id.isin(CONTRASTS) & manifest.aligned].copy()
    indexed = []
    for frame in (reference, arm):
        require(not frame.prompt_uid.isna().any() and not frame.prompt_uid.duplicated().any(),
                "missing or duplicate margin prompt_uid")
        require(np.isfinite(frame.margin_fp32).all(), "nonfinite margin")
        indexed.append(frame.set_index("prompt_uid"))
    ref, model = indexed
    require(set(ref.index) == set(model.index), "margin prompt sets differ")
    for field in ("candidate_id", "condition"):
        require(ref[field].equals(model[field].reindex(ref.index)), f"margin {field} mismatch")
    for name, frame in (("ref", ref), ("arm", model)):
        for side in ("a", "b"):
            uid = rows[f"uid_{side}"]
            require(uid.notna().all() and uid.isin(frame.index).all(), "missing contrast prompt")
            require(uid.map(frame.candidate_id).equals(rows.candidate_id), "contrast candidate mismatch")
        rows[f"abs_dm_{name}"] = abs(rows.uid_a.map(frame.margin_fp32) - rows.uid_b.map(frame.margin_fp32))
    free = ref[ref.condition.eq("attr_free")]
    free_model = model.reindex(free.index)
    scale = float(free_model.margin_fp32.abs().median() / free.margin_fp32.abs().median())
    result = []
    for contrast in CONTRASTS:
        selected = rows[rows.contrast_id.eq(contrast)]
        require(len(selected) == 450 and selected.candidate_id.nunique() == 150,
                f"full-grid coverage differs: {contrast}")
        row = {"contrast": contrast, "pairs": len(selected), "candidates": selected.candidate_id.nunique(),
               "r_scale": scale, "scope": "all aligned pairs; quantiles are descriptive, not CIs"}
        for name in ("ref", "arm"):
            values = selected[f"abs_dm_{name}"]
            row.update({f"p50_abs_effect_{name}": float(values.quantile(.5)),
                        f"p99_abs_effect_{name}": float(values.quantile(.99)),
                        f"mean_abs_effect_{name}": float(values.mean())})
        for metric in ("p50", "p99"):
            row[f"{metric}_ratio_vs_sft"] = row[f"{metric}_abs_effect_arm"] / row[f"{metric}_abs_effect_ref"]
        row["mean_ratio_vs_sft"] = row["mean_abs_effect_arm"] / row["mean_abs_effect_ref"]
        row["rho_mean_ratio_over_scale"] = row["mean_ratio_vs_sft"] / scale
        result.append(row)
    return result


def baseline_diagnostics(selected, pairs, margins):
    """Compare current-model baselines across patch batches and full-margin batches.

    This is a numerical-context diagnostic, not a test that historical adapter
    identity was correct. Report differences; do not silently treat them as noise.
    """
    index = margins.set_index("prompt_uid").margin_fp32
    linked = selected[IDENTITY].merge(pairs, on=["pair_uid", "candidate_id"], validate="one_to_one")
    patch = np.stack([linked.m_a, linked.m_b], axis=1).astype(np.float64)
    full = np.stack([linked.uid_a.map(index), linked.uid_b.map(index)], axis=1).astype(np.float64)
    require(np.isfinite(full).all() and np.isfinite(patch).all(), "nonfinite linked baseline")
    error = abs(patch - full).ravel()
    dm_patch, dm_full = patch[:, 0] - patch[:, 1], full[:, 0] - full[:, 1]
    return {"max_patch_vs_full_margin_abs_difference": float(error.max()),
            "p50_patch_vs_full_margin_abs_difference": float(np.quantile(error, .5)),
            "p99_patch_vs_full_margin_abs_difference": float(np.quantile(error, .99)),
            "mean_patch_vs_full_margin_abs_difference": float(error.mean()),
            "patch_vs_full_decision_disagreements": int(((patch > 0) != (full > 0)).sum()),
            "compared_prompt_occurrences": int(error.size),
            "patch_vs_full_dm_sign_disagreements": int((np.sign(dm_patch) != np.sign(dm_full)).sum()),
            "patch_vs_full_dm_pearson_r": float(np.corrcoef(dm_patch, dm_full)[0, 1]),
            "selected_mean_abs_dm_patch": float(abs(dm_patch).mean()),
            "selected_mean_abs_dm_full": float(abs(dm_full).mean()),
            "selected_p99_abs_dm_patch": float(np.quantile(abs(dm_patch), .99)),
            "selected_p99_abs_dm_full": float(np.quantile(abs(dm_full), .99))}


def read_json(path):
    return json.loads(path.read_text())


def json_clean(value):
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_clean(v) for v in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    inputs = (args.input_root or Path(os.environ.get("HBM_OUTPUT_ROOT", root.parent))).resolve()
    out = args.out or root / "reports/research_followup/mechanism_seeds"
    inter = inputs / "outputs/interp"
    reports = root / "reports/interp"
    manifest_path = inter / "manifest/contrasts-grid-en.parquet"
    if args.inventory:
        files = [manifest_path]
        for spec in SPECS:
            files.append(inter / "margins" / spec / "grid-en.parquet")
            for contrast in CONTRASTS:
                rep = reports / "patching" / f"{spec}-grid-{contrast}{SUFFIX}"
                files.extend(rep / f"{part}.csv" for part in ("pairs", "real", "atp", "verify"))
        print(json.dumps({"files": len(files), "missing": [str(p) for p in files if not p.exists()],
                          "bytes": sum(p.stat().st_size for p in files if p.exists())}))
        return
    os.environ["HBM_OUTPUT_ROOT"] = str(inputs)
    sys.path.insert(0, str(root / "src"))
    from hiring_bias_mitigation.interp.spec import load_spec_set, resolve_adapter
    models = load_spec_set(root / "configs/interp/models.yaml")
    required_adapters = sorted({a for spec in SPECS for a in models.specs[spec].adapters})
    adapters = {}
    for name in required_adapters:
        path = Path(resolve_adapter(models.adapter_paths[name])) / "adapter_model.safetensors"
        adapters[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
    expected_fp = {s: {a: adapters[a]["sha256"][:16] for a in models.specs[s].adapters} for s in SPECS}
    paths = set()
    def track(path):
        require(path.exists(), f"missing input {path}")
        paths.add(path)
        return path
    track(Path(__file__))
    track(root / "configs/interp/models.yaml")
    canonical_path = track(root / "scripts/interp/10_patch.py")
    track(root / "scripts/interp/20_distill_compare.py")
    track(root / "src/hiring_bias_mitigation/interp/patching.py")
    for source in ("scripts/interp/02_margins.py", "src/hiring_bias_mitigation/interp/margins.py",
                   "src/hiring_bias_mitigation/interp/render.py", "src/hiring_bias_mitigation/interp/segment.py"):
        track(root / source)
    helpers = load_pure_functions(canonical_path, ("select_pairs", "load_completed"))
    manifest = pd.read_parquet(track(manifest_path))
    track(Path(str(manifest_path) + ".meta.json"))
    require(not manifest.pair_uid.duplicated().any(), "duplicate manifest pair_uid")
    canonical = {c: helpers["select_pairs"](manifest, c, 64, .5, 42, "sft_v3") for c in CONTRASTS}
    selected_uids = {uid for frame in canonical.values() for side in ("uid_a", "uid_b") for uid in frame[side]}
    segments = pd.read_parquet(track(inter / "segments/grid-en.parquet"),
                               columns=["prompt_uid", "input_ids", "n_tokens"])
    segments = segments[segments.prompt_uid.isin(selected_uids)]
    require(len(segments) == len(selected_uids) and segments.prompt_uid.nunique() == len(selected_uids),
            "selected segment prompt coverage differs")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(models.base_model, cache_dir=str(inputs / "hf-cache/hub"),
                                              local_files_only=True)
    decoded = tokenizer.batch_decode(segments.input_ids.tolist(), skip_special_tokens=False,
                                     clean_up_tokenization_spaces=False)
    token_uid_matches = [hashlib.sha1(text.encode()).hexdigest()[:16] == uid
                         for text, uid in zip(decoded, segments.prompt_uid)]
    require(all(token_uid_matches), "selected segmentation tokens decode to different prompt identities")
    require(all(len(ids) == n for ids, n in zip(segments.input_ids, segments.n_tokens)), "segment token length mismatch")
    del decoded  # Private prompt strings are never exported.
    # A decode hash alone would allow alternate tokenizations of identical text.
    # Rebuild the actual grid with the recorded full-margin config and compare IDs.
    from hiring_bias_mitigation.interp import render
    ref_config = read_json(inter / "margins/sft_v3/grid-en.parquet.meta.json")["data_config"]
    rendered = render.render(render.build_frame("grid", ref_config), tokenizer,
                             ref_config.get("chat_template_kwargs"))
    require(len(rendered) == 31050 and rendered.prompt_uid.nunique() == 31050, "re-rendered full grid differs")
    rendered_uids = set(rendered.prompt_uid)
    text_by_uid = rendered.set_index("prompt_uid").text
    require(selected_uids.issubset(rendered_uids), "selected prompt absent from current re-rendered grid")
    encoded = tokenizer(segments.prompt_uid.map(text_by_uid).tolist(), add_special_tokens=False)["input_ids"]
    require(all(np.array_equal(old, new) for old, new in zip(segments.input_ids, encoded)),
            "cached segmentation tokens differ from current grid render/tokenization")
    del rendered, text_by_uid, encoded
    markers, warnings, margin_checks, margins, patch, checks, raw_checks = {}, [], [], {}, {}, [], []
    profiles, targets, verified, tails = [], [], [], []
    for spec in SPECS:
        marker = root / "reports/experiment_followup" / f"mechanism_{spec}.json"
        if marker.exists():
            mark = read_json(track(marker))
            require(mark.get("completed") is True and mark.get("spec") == spec, "invalid completion marker")
            require(mark.get("selection_reference") == "sft_v3" and mark.get("selection_seed") == 42
                    and mark.get("contrasts") == 4, "completion marker selection differs")
            markers[spec] = mark
        else:
            require(spec not in NEW_SPECS, f"new job missing completion marker: {spec}")
            markers[spec] = {"status": "legacy checkpoint; marker not used"}
        path = inter / "margins" / spec / "grid-en.parquet"
        margins[spec] = pd.read_parquet(track(path), columns=["prompt_uid", "candidate_id", "condition", "margin_fp32", "n_tokens"])
        m = margins[spec]
        require(len(m) == 31050 and m.prompt_uid.nunique() == 31050, f"incomplete margins: {spec}")
        require(set(m.prompt_uid) == rendered_uids, "full-margin prompts differ from current re-rendered grid")
        counts = segments.set_index("prompt_uid").n_tokens
        require(counts.equals(m.set_index("prompt_uid").n_tokens.reindex(counts.index)), "patch/full-margin prompt token lengths differ")
        meta = read_json(track(Path(str(path) + ".meta.json")))
        require(meta.get("adapters") == expected_fp[spec], f"margin adapter mismatch: {spec}")
        summary = read_json(track(reports / f"margins_summary-{spec}-grid-en.json"))
        audit = summary.get("audit", {})
        require(summary.get("rows") == 31050 and audit.get("available") and audit.get("rows_matched") == 31050
                and audit.get("agreement_pct", 0) >= 97, f"margin audit failed: {spec}")
        margin_checks.append({"spec": spec, "rows": len(m), "audit_agreement_pct": audit["agreement_pct"],
                              "adapters_match": True, "kernels": meta.get("kernels"),
                              "torch": meta.get("torch"), "transformers": meta.get("transformers")})
        for contrast in CONTRASTS:
            label = {"spec": spec, "method": ARMS.get(spec, ("SFT-v3", None))[0],
                     "seed": ARMS.get(spec, (None, None))[1], "contrast": contrast}
            cache = inter / "patching" / spec / "grid" / f"{contrast}{SUFFIX}"
            rep = reports / "patching" / f"{spec}-grid-{contrast}{SUFFIX}"
            selected = pd.read_parquet(track(cache / "selected_pairs.parquet"))
            expected = canonical[contrast]
            same_identity(expected, selected)
            require(len(selected) == (45 if contrast == "gen_male_female_impl" else 64), "unexpected selected count")
            run = read_json(track(cache / "run.meta.json"))
            require(run.get("spec") == spec and run.get("contrast") == contrast
                    and run.get("select_on") == "sft_v3" and run.get("seed") == 42
                    and run.get("min_effect") == .5 and run.get("pairs") == len(selected),
                    f"run selection metadata differs: {spec}/{contrast}")
            identity_path = cache / "poster_followup.meta.json"
            if identity_path.exists():
                identity = read_json(track(identity_path))
                require(identity.get("adapters") == expected_fp[spec], "patch identity fingerprint mismatch")
                identity_status = identity.get("prior_cache_provenance")
            else:
                require(spec not in NEW_SPECS, "new patch has no bound identity")
                identity_status = "legacy_cache_without_adapter_identity_metadata"
            if identity_status != "empty_cache_bound_before_first_compute":
                warnings.append(f"{spec}/{contrast}: {identity_status}; current fingerprints do not prove historical cache identity")
            tables = {part: pd.read_csv(track(rep / f"{part}.csv")) for part in ("pairs", "real", "atp", "verify")}
            same_identity(selected, tables["pairs"], ["pair_uid", "candidate_id"])
            raw = helpers["load_completed"](cache)
            # Canonical summarise_run represents the final pre-norm component by -1.
            raw["real"]["layer"] = raw["real"]["layer"].fillna(-1).astype(int)
            same_identity(tables["pairs"], raw["pairs"], ["pair_uid", "candidate_id"])
            for batch in cache.glob("pairs-*.parquet"):
                tag = batch.name[len("pairs-"):]
                if all((cache / f"{part}-{tag}").exists() for part in ("real", "atp")):
                    for part in ("pairs", "real", "atp"):
                        track(cache / f"{part}-{tag}")
            for field in ("m_a", "m_b"):
                merged = tables["pairs"].set_index("pair_uid")[field].sort_index()
                original = raw["pairs"].set_index("pair_uid")[field].sort_index()
                # pandas' float32 CSV writer emits its shortest decimal representation.
                require(np.allclose(merged, original, rtol=1e-7, atol=1e-6), "pair CSV baseline differs from raw")
            raw_verify = pd.read_parquet(track(cache / "verify.parquet"))
            for part, keys, frame in (("real", ["kind", "layer", "cls"], raw["real"]),
                                     ("atp", ["kind", "layer", "head", "cls"], raw["atp"]),
                                     ("verify", ["layer", "head", "cls", "control"], raw_verify)):
                recomputed = effect_summary(raw["pairs"], frame, keys, atp=part == "atp")
                try:
                    errors = compare_summary(recomputed, tables[part], keys)
                except ValueError as exc:
                    raise ValueError(f"{spec}/{contrast}/{part}: {exc}") from exc
                raw_checks.append({**label, "part": part, "summary_groups": len(recomputed),
                                   "max_effect_abs_error": errors["removed_logits_per_pair"],
                                   "max_share_abs_error": errors["share_removed"]})
            diagnostic = baseline_diagnostics(selected, tables["pairs"], m)
            # A complete final-pre-norm swap should read B's own margin. This is
            # an internal intervention-path control, separate from cross-pass drift.
            final_swap = raw["real"][raw["real"].kind.eq("final_prenorm") & raw["real"].cls.eq("suffix")]
            final_swap = final_swap.merge(raw["pairs"][["pair_uid", "m_b"]], on="pair_uid", validate="one_to_one")
            require(len(final_swap) == len(selected), "missing final-pre-norm control")
            diagnostic["max_final_prenorm_swap_vs_b_abs_difference"] = float(abs(final_swap.m_patched - final_swap.m_b).max())
            checks.append({**label, "pairs": len(selected), "selection_hash": semantic_hash(selected, IDENTITY),
                           "matches_canonical_sft_selection": True, "selection_reference": "sft_v3",
                           "selection_seed": 42, "min_abs_selection_effect": .5,
                           "cache_identity_status": identity_status, **diagnostic})
            patch[(spec, contrast)] = tables
            rows, overview = head_tables(tables["verify"], tables["atp"])
            targets.extend({**label, **r} for r in rows)
            verified.append({**label, **overview})
        print(f"checked {spec}: 31,050 margins, four fixed selections, raw/CSV agreement", flush=True)
    # Runtime comparability is checked on actual inference metadata, not CPU manifest metadata.
    for record in margin_checks[1:]:
        for key in ("kernels", "torch", "transformers"):
            require(record[key] == margin_checks[0][key], f"margin runtime mismatch: {key}")
    for spec, (method, seed) in ARMS.items():
        label = {"spec": spec, "method": method, "seed": seed}
        saved = pd.read_csv(track(reports / "distill" / spec / "component_summary.csv")).set_index("contrast")
        for contrast in CONTRASTS:
            p = component_profile(patch[("sft_v3", contrast)]["real"], patch[(spec, contrast)]["real"])
            require(np.isclose(p["real_component_pearson_r"], saved.loc[contrast, "profile_r"], rtol=1e-7, atol=1e-8),
                    "recomputed component correlation differs from canonical report")
            profiles.append({**label, "contrast": contrast, "pairs": len(canonical[contrast]), **p})
        arm_tails = full_grid_effects(manifest, margins["sft_v3"], margins[spec])
        saved_tails = pd.read_csv(track(reports / "distill" / spec / "sensitivity.csv")).set_index("contrast")
        for row in arm_tails:
            for ours, theirs in (("p50_ratio_vs_sft", "r_attr_median"), ("p99_ratio_vs_sft", "r_attr_p99"),
                                 ("mean_ratio_vs_sft", "r_attr"), ("r_scale", "r_scale")):
                # The canonical mean uses candidate-grouped float32 sums; this pass
                # also checks quantiles directly. Sub-micro-unit reduction error is expected.
                row[f"canonical_abs_error_{ours}"] = abs(row[ours] - saved_tails.loc[row["contrast"], theirs])
                require(np.isclose(row[ours], saved_tails.loc[row["contrast"], theirs], rtol=1e-6, atol=1e-7),
                        f"recomputed full-grid effect differs from canonical report: {spec}/{row['contrast']}/{ours}")
            tails.append({**label, **row})
    pframe, tframe, hframe = pd.DataFrame(profiles), pd.DataFrame(tails), pd.DataFrame(targets)
    check_frame = pd.DataFrame(checks)
    method_summary = {}
    for method in dict.fromkeys(m for m, _ in ARMS.values()):
        p, t = pframe[pframe.method.eq(method)], tframe[tframe.method.eq(method)]
        h = hframe[hframe.method.eq(method)]
        method_summary[method] = {
            "seeds": [42, 43, 44], "contrast_seed_cells": len(t),
            "real_component_pearson_r_range": [p.real_component_pearson_r.min(), p.real_component_pearson_r.max()],
            "p50_ratio_range": [t.p50_ratio_vs_sft.min(), t.p50_ratio_vs_sft.max()],
            "p99_ratio_range": [t.p99_ratio_vs_sft.min(), t.p99_ratio_vs_sft.max()],
            "p50_down_p99_up_cells": int(((t.p50_ratio_vs_sft < 1) & (t.p99_ratio_vs_sft > 1)).sum()),
            "p50_down_p99_down_cells": int(((t.p50_ratio_vs_sft < 1) & (t.p99_ratio_vs_sft < 1)).sum()),
            "targets": {target: {"verified_cells": int(g.verified.sum()),
                "rank_one_tested_cells": int(g.real_rank_tested.eq(1).sum()),
                "tested_rank_range": [g.real_rank_tested.min(), g.real_rank_tested.max()],
                "real_effect_range": [g.real_removed_logits_per_pair.min(), g.real_removed_logits_per_pair.max()]}
                for target, g in h.groupby("target")},
        }
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
        "methods": method_summary, "validated": {"new_completion_markers": len(NEW_SPECS),
            "margins": len(margin_checks), "fixed_selections": len(checks),
            "raw_summary_reconciliations": len(raw_checks), "real_profiles": len(profiles), "full_grid_tail_cells": len(tails),
            "selected_prompt_tokens_decode_to_uid": len(selected_uids),
            "selected_prompt_tokens_equal_fresh_grid_render": len(selected_uids),
            "patch_caches_bound_before_compute": int(check_frame.cache_identity_status.eq("empty_cache_bound_before_first_compute").sum()),
            "patch_caches_historical_identity_unverified": len(warnings)},
        "patch_vs_full_margin_diagnostics": {
            "maximum_absolute_difference": check_frame.max_patch_vs_full_margin_abs_difference.max(),
            "max_internal_final_prenorm_swap_control_error": check_frame.max_final_prenorm_swap_vs_b_abs_difference.max(),
            "method_ranges": {method: {
                "max_abs_difference_range": [g.max_patch_vs_full_margin_abs_difference.min(), g.max_patch_vs_full_margin_abs_difference.max()],
                "median_abs_difference_range": [g.p50_patch_vs_full_margin_abs_difference.min(), g.p50_patch_vs_full_margin_abs_difference.max()],
                "dm_sign_disagreements_sum": int(g.patch_vs_full_dm_sign_disagreements.sum()),
                "compared_pairs_sum": int(g.pairs.sum())} for method, g in check_frame.groupby("method")},
            "interpretation": "Patch baselines are live current-arm forwards, not stored SFT selection values. Matching hashed prompt/token identities and current adapter identities rules out those semantic mismatches for newly bound caches. Patch passes use token-budgeted pair batches; full margins use batches of eight. The observed numerical-context differences also occur in fresh caches. Existing CPU artifacts do not isolate their cause or establish a numerical-noise floor for intervention effects."},
        "scope": {"reference": "one shared SFT-v3 checkpoint",
            "selection": "Canonical SFT-v3 |effect| >= 0.5; seed 42; 64 pairs, gender 45; identical across all arms/seeds",
            "component_profile": "Pearson across 64 real mixer/MLP suffix-patching effects",
            "head_rank": "Descending signed real effect only among each run's AtP-selected plus control tested heads; not all-model rank",
            "head_ci": "Existing candidate-bootstrap 95% CI is on share_removed, not logits per pair",
            "tail": "p50/p99 of absolute logit effects across 450 aligned CV-job pairs per contrast; ratios to shared SFT; descriptive ranges, not CIs",
            "execution": "HF interpretation loads the original Base and activates [SFT-v3, last-stage] adapters together; behavioral serving uses successively merged weights. Algebraic adapter addition is not evidence of numerical equivalence across these execution paths",
            "batch_context": "Patch baselines and full-grid margins differ under their different batching contexts; see patch_vs_full_margin_diagnostics. Do not pool their per-pair margins as numerically interchangeable measurements",
            "inference": "Replicated checkpoints on one development benchmark do not establish causal routes or external generalization. Neither p99 effects nor L23.H1 results identify the cause of high-hire-rate behavioral deterioration"},
        "warnings": warnings}
    provenance = {"inputs": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(paths)],
                  "adapters": adapters, "expected_adapter_fingerprints": expected_fp,
                  "completion_markers": markers, "margin_checks": margin_checks,
                  "execution_scope": summary["scope"]["execution"],
                  "warnings": warnings}
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in {"component_profiles": pframe, "target_heads": hframe,
                        "head_verification": pd.DataFrame(verified), "tail_effects": tframe,
                        "selection_checks": check_frame, "raw_summary_checks": pd.DataFrame(raw_checks)}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    for name, value in (("summary", summary), ("provenance", provenance)):
        (out / f"{name}.json").write_text(json.dumps(json_clean(value), indent=2, allow_nan=False) + "\n")
    lines = ["# Three-seed mechanism follow-up", "", "All six added jobs passed their completion and artifact checks.", "",
        "| Method | Real component Pearson r | p50 ratio | p99 ratio | p50 down, p99 up |", "|---|---:|---:|---:|---:|"]
    for method, values in method_summary.items():
        ranges = [f"{values[k][0]:.3f} to {values[k][1]:.3f}" for k in ("real_component_pearson_r_range", "p50_ratio_range", "p99_ratio_range")]
        lines.append(f"| {method} | " + " | ".join(ranges) + f" | {values['p50_down_p99_up_cells']}/12 |")
    lines += ["", "L15.H11 ranks first among tested heads in 5/12 DPO, 12/12 GRPO and 11/12 self-distillation cells. L23.H1 ranks first in only 2/12 DPO cells. These are ranks within each adaptively selected test set, not ranks across the model.",
              "", "For DPO combat/implicit, L23.H1 has true-intervention ranks 1/46, 2/47 and 2/44 for seeds 42, 43 and 44. A claim that it is always the new leading head is not supported."]
    lines += ["", "Each range spans three seeds and four contrasts. These are descriptive ranges, not confidence intervals.",
        "", "## Interpretation and limits", "", *[f"- {value}." for value in summary["scope"].values()],
        "- AtP columns remain explicitly separate from real intervention columns. The tested head set changes by arm and seed.",
        f"- Patch versus full-grid baselines differ by up to {summary['patch_vs_full_margin_diagnostics']['maximum_absolute_difference']:.2f} logits, including fresh bound caches. Selected tokens decode to the same hashed prompt strings. The two passes use different batch construction. This is a measured execution-context discrepancy, not proof that the difference is harmless noise. Do not substitute full-grid baselines into patch-effect denominators.",
        f"- The internal final-pre-norm swap control differs from B's recorded margin by at most {summary['patch_vs_full_margin_diagnostics']['max_internal_final_prenorm_swap_control_error']:.6g} logits. This checks the intervention readout path; it does not resolve the separate cross-pass discrepancy.",
        "- The old SFT-v3, DPO seed 42, GRPO seed 42 and self-distillation seed 43 patch caches lack historical adapter-identity metadata. Self-distillation seed 42 reused legacy combat/implicit and religion/implicit caches; its other two contrasts were bound before new computation. Thus 18/40 caches retain a historical identity limitation and 22/40 were bound before computation. All current full-grid margin fingerprints match current weights. Raw effects reconcile with exported CSVs, and every pair selection matches, but those checks cannot prove historical cache identity.",
        "", "## Reproduce", "", "```sh", f"{sys.executable} scripts/summarize_mechanism_seeds.py --input-root {inputs}", "```", "",
        "The script writes only this aggregate report directory. Source-file SHA256 values and current adapter hashes are recorded in provenance.json. No CV text or candidate identifiers are exported.",
        "", "## Files", "", "- component_profiles.csv: 36 genuine intervention profile correlations, cross-checked against canonical component_summary.csv.",
        "- target_heads.csv: L15.H11 and L23.H1, real effects, share CIs, ranks within tested heads, and separate AtP estimates.",
        "- head_verification.csv: tested counts, controls, top tested heads and subset-only AtP agreement.",
        "- tail_effects.csv: 36 all-pair distributions, recomputed from current margin files and cross-checked against sensitivity.csv.",
        "- selection_checks.csv and raw_summary_checks.csv: canonical selection and raw-to-CSV reconciliation.",
        "- summary.json and provenance.json: aggregate findings, scope, hashes and historical identity limitations."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(json_clean(summary), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
