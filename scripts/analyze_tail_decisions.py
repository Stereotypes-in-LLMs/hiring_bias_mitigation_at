"""Read-only CPU analysis of tail effects, threshold crossings, and affected units.

Existing vLLM op-margin-v2 and HF margin_fp32 scores are analyzed separately. Each
model/backend gets its own cut on the complete 31,050-prompt grid. No inference,
training, bootstrapping, model selection, or raw CV/JD text export is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import poster_audit_followup as P

ROOT = Path(__file__).resolve().parents[1]
CONTRASTS = {
    "mil_combat_civ_impl": ("military_status", "implicit", "Participant in combat actions", "Civilian"),
    "mil_combat_civ_expl": ("military_status", "explicit", "Participant in combat actions", "Civilian"),
    "rel_atheist_christian_impl": ("religion", "implicit", "Atheist", "Christian"),
    "gen_male_female_impl": ("gender", "implicit", "Male", "Female"),
}
SPECS = {
    "SFT-v3": ["sft_v3"],
    "DPO": ["dpo4880", "dpo4880_s43", "dpo4880_s44"],
    "GRPO + lr decay": ["grpo_r4", "grpo_r4_s43", "grpo_r4_s44"],
    "Self-distillation": ["distill42", "distill43", "distill44"],
}


def opaque(kind: str, value: str) -> str:
    return hashlib.sha256(f"decision-geometry-v1:{kind}:{value}".encode()).hexdigest()[:20]


def validate_frame(frame: pd.DataFrame, keys: list[str], reference=None) -> pd.DataFrame:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("missing or duplicate prompt key")
    if not np.isfinite(frame.margin).all():
        raise ValueError("nonfinite margin")
    frame = frame.sort_values(keys).reset_index(drop=True).copy()
    if reference is not None and not frame[keys].equals(reference[keys]):
        raise ValueError("prompt keys differ from reference")
    ids = frame.pair_id.astype(str).str.split("__")
    if not ids.map(len).eq(2).all():
        raise ValueError("pair_id must encode exactly candidate__job")
    for column, part in (("candidate_id", 0), ("job_id", 1)):
        parsed = ids.str[part]
        if column in frame and not np.array_equal(frame[column].to_numpy(dtype=str), parsed.to_numpy(dtype=str)):
            raise ValueError(f"{column} disagrees with pair_id")
        frame[column] = parsed
    return frame


def calibration(margins, count: int, opc) -> dict:
    m = np.asarray(margins, dtype=float)
    if not 0 <= count <= len(m):
        raise ValueError("hire count outside grid")
    threshold, weight = opc.cut_for_count(m, count)
    actual = (m > threshold).sum() + weight * (m == threshold).sum()
    if not 0 <= weight <= 1 or not np.isclose(actual, count):
        raise ValueError("cut fails requested hire count")
    return {"threshold": threshold, "tie_weight": weight,
            "tied_prompts": int((m == threshold).sum()), "hire_count": count,
            "actual_hire_rate_pct": 100 * actual / len(m)}


def interpolate(a, b, weight):
    return (1 - weight) * a.astype(float) + weight * b.astype(float)


def memberships(frame: pd.DataFrame, cut: dict, opc) -> dict:
    """Aggregate each strict/inclusive branch BEFORE interpolating tied cuts.

    Taking max of already-fractional set flags would undercount a pair when one
    of its sets is unstable at the strict cut and another at the inclusive cut.
    """
    branches = []
    for hire in (frame.m.to_numpy() > cut["threshold"],
                 frame.m.to_numpy() >= cut["threshold"]):
        sets = opc._set_flags(frame, hire)
        pair = sets.groupby(level="pair_id").max()
        cv = pair.groupby(pair.index.str.split("__").str[0]).max()
        cv.index.name = "candidate_id"
        branches.append({"set": sets, "pair": pair, "cv": cv})
    return {unit: interpolate(branches[0][unit], branches[1][unit], cut["tie_weight"])
            for unit in branches[0]}


def crossing(a, b, cut):
    a, b = np.asarray(a), np.asarray(b)
    strict = (a > cut["threshold"]) != (b > cut["threshold"])
    inclusive = (a >= cut["threshold"]) != (b >= cut["threshold"])
    return interpolate(strict, inclusive, cut["tie_weight"])


def pair_features(frame: pd.DataFrame, contrast: str) -> pd.DataFrame:
    family, cue, a, b = CONTRASTS[contrast]
    selected = frame[(frame.protected_group == family) & (frame.condition == cue)]
    values = selected.pivot(index="pair_id", columns="protected_attr", values="m")
    raw = selected.pivot(index="pair_id", columns="protected_attr", values="margin")
    ranks = selected.pivot(index="pair_id", columns="protected_attr", values="percentile")
    if a not in values or b not in values or values[[a, b]].isna().any().any():
        raise ValueError(f"incomplete contrast: {contrast}")
    free = frame[frame.condition == "attr_free"].set_index("pair_id").m
    result = pd.DataFrame({"a": values[a], "b": values[b],
                           "absolute_effect": (raw[a] - raw[b]).abs(),
                           "rank_effect": (ranks[a] - ranks[b]).abs(),
                           "direction": np.sign(raw[a] - raw[b]),
                           "free_margin": free.reindex(values.index)})
    if result.isna().any().any():
        raise ValueError("missing attribute-free reference or effect")
    return result


def finite_ratio(a, b):
    return float(a / b) if b != 0 else None


def effect_summary(features: pd.DataFrame, ref: pd.DataFrame) -> dict:
    out = {"pairs": len(features),
           "effect_rank_spearman_vs_sft": features.absolute_effect.corr(ref.absolute_effect, method="spearman"),
           "direction_disagreement_vs_sft": float((features.direction != ref.direction).mean())}
    for col in ("absolute_effect", "rank_effect"):
        value = features[col]
        for q, name in ((.5, "p50"), (.95, "p95"), (.99, "p99")):
            out[f"{col}_{name}"] = float(value.quantile(q))
            out[f"{col}_{name}_ratio_vs_sft"] = finite_ratio(value.quantile(q), ref[col].quantile(q))
        top = value >= value.quantile(.99)
        out[f"{col}_top1pct_pairs_including_ties"] = int(top.sum())
        out[f"{col}_top1pct_share_of_total_effect"] = finite_ratio(value[top].sum(), value.sum())
    return out


def leave_one_out(delta: pd.Series, kind: str) -> pd.DataFrame:
    """Fixed-cut deletion influence; not a CI or an independently held-out test."""
    pairs = delta.index.get_level_values("pair_id")
    clusters = pairs.str.split("__").str[0 if kind == "cv" else 1]
    per = delta.groupby(clusters).agg(["sum", "count"])
    if (per["count"] == len(delta)).any():
        raise ValueError("one deletion removes all sets")
    baseline = 100 * delta.mean()
    per["delta_pp_after_removal"] = 100 * (delta.sum() - per["sum"]) / (len(delta) - per["count"])
    per["change_from_full_pp"] = per.delta_pp_after_removal - baseline
    per["full_delta_pp"] = baseline
    per["cluster_hash"] = [opaque(kind, str(x)) for x in per.index]
    return per.rename(columns={"count": "removed_sets", "sum": "removed_delta_mass"}).reset_index(drop=True)


def burden_rows(flags: dict, border: set[str]) -> pd.DataFrame:
    sets = flags["set"]
    pair = pd.DataFrame({"affected_weight": flags["pair"],
                         "expected_unstable_sets": sets.groupby(level="pair_id").sum()})
    rows = []
    for pair_id, row in pair.iterrows():
        candidate, job = pair_id.split("__")
        rows.append({"unit": "pair", "unit_hash": opaque("pair", pair_id),
                     "candidate_hash": opaque("cv", candidate), "job_hash": opaque("job", job),
                     "affected_weight": row.affected_weight,
                     "expected_unstable_sets": row.expected_unstable_sets,
                     "expected_affected_pairs": row.affected_weight,
                     "borderline_pair_count": int(pair_id in border)})
    for cv, affected in flags["cv"].items():
        ids = pair.index[pair.index.str.split("__").str[0] == cv]
        rows.append({"unit": "cv", "unit_hash": opaque("cv", cv),
                     "candidate_hash": opaque("cv", cv), "job_hash": "",
                     "affected_weight": affected,
                     "expected_unstable_sets": pair.loc[ids, "expected_unstable_sets"].sum(),
                     "expected_affected_pairs": pair.loc[ids, "affected_weight"].sum(),
                     "borderline_pair_count": len(set(ids) & border)})
    return pd.DataFrame(rows)


def load_inputs(input_root: Path, opc, backends: list[str]):
    models = [("SFT-v3", 0, P.REF, "sft_v3")]
    models += [(recipe, seed, run, spec) for recipe, runs in P.RECIPES.items()
               for seed, run, spec in zip((42, 43, 44), runs, SPECS[recipe])]
    frames, provenance = {}, []
    canonical, meta_ref = P.load_scores(input_root, P.REF, opc)
    canonical = validate_frame(canonical, opc.KEY)
    if len(canonical) != 31050 or canonical.pair_id.nunique() != 450:
        raise ValueError("expected complete 31,050-prompt / 450-pair benchmark")
    common_count = int((opc.snap(canonical.margin) > 0).sum())
    free = canonical[canonical.condition == "attr_free"].set_index("pair_id").margin
    border = set(free.index[free.abs() < 2])
    for backend in backends:
        prompt_uids = None
        for recipe, seed, run, spec in models:
            if backend == "vllm":
                frame, meta = P.load_scores(input_root, run, opc)
                opc.check_comparable(meta_ref, meta)
                path, meta_path = P.score_paths(input_root, run)
            else:
                path = input_root / "outputs/interp/margins" / spec / "grid-en.parquet"
                meta_path = Path(str(path) + ".meta.json")
                meta = json.loads(meta_path.read_text())
                frame = pd.read_parquet(path, columns=[*opc.KEY, "candidate_id", "job_id",
                                                       "prompt_uid", "margin_fp32"])
                frame = frame.rename(columns={"margin_fp32": "margin"})
                if frame.prompt_uid.isna().any() or frame.prompt_uid.duplicated().any():
                    raise ValueError("HF prompt_uid missing or duplicated")
                uid_keys = frame.set_index(opc.KEY).prompt_uid.sort_index()
                if prompt_uids is not None and not uid_keys.equals(prompt_uids):
                    raise ValueError("HF prompt_uid/key correspondence changed")
                prompt_uids = uid_keys
            frame = validate_frame(frame, opc.KEY, canonical)
            frame["m"] = opc.snap(frame.margin) if backend == "vllm" else frame.margin.to_numpy()
            # Midrank empirical percentiles on the complete model/backend grid:
            # invariant to strictly increasing transforms that preserve ties.
            frame["percentile"] = frame.m.rank(method="average") / len(frame)
            frames[(backend, recipe, seed)] = frame
            provenance.append({"backend": backend, "recipe": recipe, "seed": seed,
                               "run": run, "spec": spec, "path": str(path),
                               "sha256": P.digest(path), "metadata_path": str(meta_path),
                               "metadata_sha256": P.digest(meta_path),
                               "metadata": meta})
    return frames, provenance, common_count, border


def analyze(input_root: Path, out: Path, backends: list[str]):
    opc = P.load_matching()
    frames, sources, common_count, border = load_inputs(input_root, opc, backends)
    n = len(next(iter(frames.values())))
    # Exact final_table.py grid and rounding, including its half-to-even cases.
    rates = [(f"{100 * h:g}", round(h * n))
             for h in np.round(np.arange(.10, .3501, .005), 3)]
    rates += [("sft_common", common_count)]
    cuts, effects, geometry, global_rows, detail, overlap, burden, influence, groups, high_pairs = ([] for _ in range(10))
    cached = {}
    for backend in backends:
        ref = frames[(backend, "SFT-v3", 0)]
        ref_features = {c: pair_features(ref, c) for c in CONTRASTS}
        for rate, count in rates:
            cut = calibration(ref.m, count, opc)
            cached[(backend, rate)] = (cut, memberships(ref, cut, opc))
        for (be, recipe, seed), frame in frames.items():
            if be != backend:
                continue
            print(f"analyzing {backend}: {recipe} seed={seed}", flush=True)
            identity = {"backend": backend, "recipe": recipe, "seed": seed}
            features = {c: pair_features(frame, c) for c in CONTRASTS}
            for contrast, f in features.items():
                effects.append({**identity, "contrast": contrast,
                                **effect_summary(f, ref_features[contrast])})
            for rate, count in rates:
                ref_cut, ref_flags = cached[(backend, rate)]
                cut = calibration(frame.m, count, opc)
                flags = memberships(frame, cut, opc)
                if backend == "vllm":
                    expected, _ = opc.matched_flags(frame, "margin", count)
                    if not np.allclose(flags["set"], expected):
                        raise ValueError("set matching differs from canonical")
                id_rate = {**identity, "rate_label": rate,
                           "actual_hire_rate_pct": cut["actual_hire_rate_pct"]}
                cuts.append({**id_rate, **cut})
                is_border = flags["set"].index.get_level_values("pair_id").isin(border)
                delta = flags["set"] - ref_flags["set"]
                global_rows.append({**id_rate,
                    "unstable_set_pct": 100 * flags["set"].mean(),
                    "sft_unstable_set_pct": 100 * ref_flags["set"].mean(),
                    "delta_unstable_set_pp": 100 * delta.mean(),
                    "affected_pair_weight": flags["pair"].sum(),
                    "affected_cv_weight": flags["cv"].sum(),
                    "unstable_sets_on_35_borderline_share": finite_ratio(flags["set"][is_border].sum(), flags["set"].sum()),
                    "delta_outside_35_pp": 100 * delta[~is_border].mean(),
                    "delta_inside_35_pp": 100 * delta[is_border].mean()})
                group_total = 0.
                for (family, cue), d in delta.groupby(level=["protected_group", "condition"]):
                    arm_group = flags["set"].reindex(d.index)
                    ref_group = ref_flags["set"].reindex(d.index)
                    group_total += d.sum()
                    groups.append({**id_rate, "family": family, "cue_style": cue,
                        "sets": len(d), "unstable_set_weight": arm_group.sum(),
                        "sft_unstable_set_weight": ref_group.sum(),
                        "delta_within_group_pp": 100 * d.mean(),
                        "contribution_to_global_delta_pp": 100 * d.sum() / len(delta),
                        "added_instability_weight_fuzzy_difference": d.clip(lower=0).sum(),
                        "removed_instability_weight_fuzzy_difference": (-d).clip(lower=0).sum()})
                if not np.isclose(group_total, delta.sum()):
                    raise ValueError("family/cue contributions fail to close")
                for contrast, f in features.items():
                    r = ref_features[contrast]
                    cross = crossing(f.a, f.b, cut)
                    ref_cross = crossing(r.a, r.b, ref_cut)
                    top = (f.absolute_effect >= f.absolute_effect.quantile(.99)).to_numpy()
                    ref_top = (r.absolute_effect >= r.absolute_effect.quantile(.99)).to_numpy()
                    extra = np.maximum(cross - ref_cross, 0)
                    removed = np.maximum(ref_cross - cross, 0)
                    midpoint = np.abs((f.a + f.b).to_numpy() / 2 - cut["threshold"])
                    near = np.minimum(np.abs(f.a - cut["threshold"]), np.abs(f.b - cut["threshold"]))
                    geometry.append({**id_rate, "contrast": contrast, "pairs": len(f),
                        "crossing_weight": float(cross.sum()), "sft_crossing_weight": float(ref_cross.sum()),
                        "delta_crossing_pp": 100 * float((cross - ref_cross).mean()),
                        "added_crossing_weight_fuzzy_difference": float(extra.sum()),
                        "removed_crossing_weight_fuzzy_difference": float(removed.sum()),
                        "top1pct_pairs_including_ties": int(top.sum()),
                        "top1pct_crossing_weight": float(cross[top].sum()),
                        "top1pct_share_all_crossings": finite_ratio(cross[top].sum(), cross.sum()),
                        "top1pct_share_added_crossings": finite_ratio(extra[top].sum(), extra.sum()),
                        "sft_top1pct_crossing_weight": float(cross[ref_top].sum()),
                        "top1pct_median_midpoint_distance_to_cut": float(np.median(midpoint[top])),
                        "top1pct_median_nearest_endpoint_distance": float(np.median(np.asarray(near)[top])),
                        "crossing_on_35_borderline_share": finite_ratio(cross[f.index.isin(border)].sum(), cross.sum())})
                    if rate == "sft_common":
                        d = f.copy()
                        d["pair_hash"] = [opaque("pair", x) for x in f.index]
                        d["candidate_hash"] = [opaque("cv", x.split("__")[0]) for x in f.index]
                        d["job_hash"] = [opaque("job", x.split("__")[1]) for x in f.index]
                        d["canonical_35_borderline"] = f.index.isin(border)
                        d["crossing_weight"] = cross
                        d["sft_crossing_weight"] = ref_cross
                        d["midpoint_distance_to_cut"] = midpoint
                        d["nearest_endpoint_distance_to_cut"] = np.asarray(near)
                        d["top1pct_including_ties"] = top
                        detail.append(d.reset_index(drop=True).assign(**id_rate, contrast=contrast))
                high_rate = rate != "sft_common" and float(rate) >= 31
                if recipe != "SFT-v3" and (rate == "sft_common" or high_rate):
                    for kind in ("cv", "job"):
                        influence.append(leave_one_out(delta, kind).assign(**id_rate, cluster_kind=kind))
                    if high_rate:
                        for pair, d in delta.groupby(level="pair_id"):
                            cv, job = pair.split("__")
                            high_pairs.append({**id_rate, "pair_hash": opaque("pair", pair),
                                "candidate_hash": opaque("cv", cv), "job_hash": opaque("job", job),
                                "canonical_35_borderline": pair in border,
                                "sets": len(d), "delta_unstable_set_mass": d.sum(),
                                "contribution_to_global_delta_pp": 100 * d.sum() / len(delta),
                                "added_instability_weight_fuzzy_difference": d.clip(lower=0).sum(),
                                "removed_instability_weight_fuzzy_difference": (-d).clip(lower=0).sum()})
                if rate == "sft_common":
                    cached[(backend, recipe, seed)] = flags
                    burden.append(burden_rows(flags, border).assign(**id_rate))
        for recipe in P.RECIPES:
            for a, b in itertools.combinations((42, 43, 44), 2):
                for unit in ("set", "pair", "cv"):
                    x = cached[(backend, recipe, a)][unit]
                    y = cached[(backend, recipe, b)][unit]
                    both, union, jac = P.weighted_jaccard(x, y)
                    overlap.append({"backend": backend, "recipe": recipe, "seeds": f"{a}/{b}",
                                    "unit": unit, "units_total": len(x),
                                    "affected_a": x.sum(), "affected_b": y.sum(),
                                    "intersection_weight": both, "union_weight": union,
                                    "weighted_jaccard": jac})
    out.mkdir(parents=True, exist_ok=True)
    tables = {"cuts": pd.DataFrame(cuts), "effect_summary": pd.DataFrame(effects),
              "threshold_geometry": pd.DataFrame(geometry), "global_curve": pd.DataFrame(global_rows),
              "common_rate_pair_features": pd.concat(detail, ignore_index=True),
              "common_rate_overlap": pd.DataFrame(overlap),
              "common_rate_burden": pd.concat(burden, ignore_index=True),
              "family_cue_curve": pd.DataFrame(groups),
              "high_rate_pair_contributions": pd.DataFrame(high_pairs),
              "leave_one_out": pd.concat(influence, ignore_index=True)}
    infl = tables["leave_one_out"]
    group = ["backend", "recipe", "seed", "rate_label", "cluster_kind"]
    tables["leave_one_out_summary"] = infl.groupby(group).agg(
        clusters=("cluster_hash", "size"), full_delta_pp=("full_delta_pp", "first"),
        min_delta_after_removal_pp=("delta_pp_after_removal", "min"),
        max_delta_after_removal_pp=("delta_pp_after_removal", "max"),
        max_abs_change_pp=("change_from_full_pp", lambda x: x.abs().max())).reset_index()
    for name, table in tables.items():
        path = out / f"{name}.csv"
        tmp = out / f".{name}.csv.tmp"
        table.to_csv(tmp, index=False)
        tmp.replace(path)
    manifest = {"analysis": "decision-geometry-v1", "complete": True,
        "backend_policy": "Separate per-model cuts; vLLM snapped to canonical 1/1024 grid; HF margin_fp32 unsnapped. No cross-backend numeric comparisons.",
        "representation_policy": "The backend labels name complete numerical paths, not a controlled backend-only comparison. Legacy HF interp.spec simultaneously activates [SFT,last] PEFT adapters on the original base. vLLM serves sequentially bf16-merged checkpoints. Differences therefore include representation/rounding as well as backend; newer experiments fixing a merged-SFT base are a different condition.",
        "common_hire_count": common_count, "common_hire_rate_pct": 100 * common_count / n,
        "grid_prompts": n,
        "pairs": int(next(iter(frames.values())).pair_id.nunique()),
        "cvs": int(next(iter(frames.values())).candidate_id.nunique()),
        "jobs": int(next(iter(frames.values())).job_id.nunique()),
        "canonical_borderline_pairs": len(border),
        "canonical_borderline_cvs": len({x.split("__")[0] for x in border}),
        "canonical_borderline_jobs": len({x.split("__")[1] for x in border}),
        "canonical_borderline_definition": "SFT-v3 vLLM attribute-free absolute margin < 2; same fixed pair membership used for both backends",
        "rate_grid": "Exact final_table.py grid: np.round(np.arange(.10,.3501,.005),3), count=round(h*31050), plus canonical SFT greedy count. Each global cut uses all 31,050 rows including 450 attribute-free prompts, as in canonical scoring.",
        "effect_definition": "Absolute difference of the two contrast margins. Rank effect is absolute difference of full-grid midrank/N; invariant to strictly increasing transforms preserving ties.",
        "tails": "p50/p95/p99 use pandas linear quantile on 450 pairs per contrast; top1% includes all pairs tied at or above empirical p99; descriptive, no CI",
        "tie_policy": "Single strict/inclusive global cut mixture; unit unions aggregated within each branch before interpolation. Seed overlap uses min/max fuzzy Jaccard, not a joint probability.",
        "crossing_difference": "Added/removed crossing weights are positive/negative membership differences, not causal transitions or probabilities under independent tie draws.",
        "scope_of_geometry": "Four binary contrasts are not the whole 2,700-set curve. family_cue_curve covers all six family/cue cells and its contributions sum to the full curve; high_rate_pair_contributions similarly closes by pair at 31-35%.",
        "leave_one_out": "Delete every set of one CV or job; keep original global cuts and ties fixed. Descriptive influence, no recalibration, bootstrap, CI, or held-out evaluation.",
        "privacy": "No CV/JD/prompt text; output unit identifiers are domain-separated SHA256 prefixes, not anonymization guarantees",
        "limitations": "Exploratory development benchmark; three fixed seeds. Rank normalization is within this grid, not a population calibration. Borderline concentration is conditioned on SFT's zero cut, especially limited at 31-35%.",
        "source_files": sources,
        "code": {str(p.relative_to(ROOT)): P.digest(p) for p in (
            Path(__file__), ROOT / "scripts/poster_audit_followup.py", ROOT / "scripts/operating_point_control.py", ROOT / "scripts/final_table.py",
            ROOT / "src/hiring_bias_mitigation/interp/spec.py", ROOT / "configs/interp/models.yaml")},
        "outputs": {name + ".csv": {"rows": len(table), "sha256": P.digest(out / f"{name}.csv")}
                    for name, table in tables.items()}}
    tmp = out / ".manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    tmp.replace(out / "manifest.json")
    print(json.dumps({"complete": True, "out": str(out), "rows": {k: len(v) for k,v in tables.items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=os.environ.get("HBM_OUTPUT_ROOT"))
    parser.add_argument("--out", type=Path, default=ROOT / "reports/research_followup/decision_geometry")
    parser.add_argument("--backends", nargs="+", choices=("vllm", "hf"), default=["vllm", "hf"])
    args = parser.parse_args()
    if args.input_root is None:
        parser.error("set --input-root or HBM_OUTPUT_ROOT")
    analyze(args.input_root, args.out, args.backends)


if __name__ == "__main__":
    main()
