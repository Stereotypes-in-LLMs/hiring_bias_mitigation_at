"""CPU checks for the poster: common-rate overlap, subgroup stability, and real head patches.

Uses existing op-margin-v2 scores and intervention outputs; never trains or scores a model.
Run with HBM_OUTPUT_ROOT set to the parent of outputs/ and the repository Python environment:

    python scripts/poster_audit_followup.py

The matching functions are compiled directly from operating_point_control.py's pure CPU
definitions. This avoids importing its GPU backends on a CPU review machine without changing
the canonical snapping, whole-set tie weights, or paired candidate-bootstrap procedure.
Every output records the canonical source hash. Missing inputs remain explicitly pending.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = "Qwen3.5-9B--en--{}--adapter--en_only_{}"
REF = RUN.format("sft", "sft_v3_additional")
RECIPES = {
    "DPO": [RUN.format("dpo", "dpo_v3_4880_step305"),
            RUN.format("dpo", "dpo_v3_4880_seed43_step305"),
            RUN.format("dpo", "dpo_v3_4880_seed44_step305")],
    "GRPO + lr decay": [RUN.format("grpo", "grpo_improve_r4_tipped_v2"),
                        RUN.format("grpo", "grpo_improve_r4_tipped_seed43"),
                        RUN.format("grpo", "grpo_improve_r4_tipped_seed44")],
    "Self-distillation": [RUN.format("sft", f"decision_distill_seed{s}")
                          for s in (42, 43, 44)],
}
SPECS = {"DPO": "dpo4880", "GRPO + lr decay": "grpo_r4",
         "Self-distillation": "distill43"}
CONTRASTS = ("mil_combat_civ_impl", "mil_combat_civ_expl",
             "rel_atheist_christian_impl", "gen_male_female_impl")
TARGET_HEADS = ((15, 11), (23, 1))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_matching(root: Path = ROOT) -> SimpleNamespace:
    """Load the canonical pure functions, not a rewritten matching implementation."""
    source = root / "scripts/operating_point_control.py"
    tree = ast.parse(source.read_text())
    constants = {"KEY", "SET", "GRID", "N_BOOT", "BOOT_SEED"}
    functions = {"snap", "_paired", "_set_flags", "unstable_share", "cut_for_count",
                 "matched_flags", "matched_comparison", "check_comparable"}
    selected, found = [], set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in constants:
                selected.append(node)
                found.add(target.id)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
            found.add(node.name)
    if found != constants | functions:
        raise RuntimeError(f"canonical matching API changed: missing {(constants | functions) - found}")
    ns = {"np": np, "pd": pd, "json": json}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    return SimpleNamespace(**ns, source_sha256=digest(source))


def score_paths(input_root: Path, run: str) -> tuple[Path, Path]:
    base = input_root / "outputs/operating_point/v2" / f"{run}--adapter"
    return Path(str(base) + ".parquet"), Path(str(base) + ".meta.json")


def load_scores(input_root: Path, run: str, opc) -> tuple[pd.DataFrame, dict]:
    path, meta_path = score_paths(input_root, run)
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("missing scores: " + str(path))
    meta = json.loads(meta_path.read_text())
    if meta.get("standard") != "op-margin-v2":
        raise ValueError(f"{meta_path}: expected op-margin-v2")
    frame = pd.read_parquet(path, columns=[*opc.KEY, "margin"])
    if frame.duplicated(opc.KEY).any() or not np.isfinite(frame["margin"]).all():
        raise ValueError(f"{path}: duplicated prompt keys or nonfinite margins")
    return frame, meta


def weighted_jaccard(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """Same fuzzy-set min/max convention as scripts/seed_overlap.py."""
    a, b = a.align(b, fill_value=0.0)
    if not a.between(0, 1).all() or not b.between(0, 1).all():
        raise ValueError("set membership must be between zero and one")
    both = float(np.minimum(a, b).sum())
    either = float(np.maximum(a, b).sum())
    return both, either, both / either if either else float("nan")


def paired_bootstrap(delta: pd.Series, n_boot: int, seed: int) -> tuple[float, float]:
    candidate = delta.index.get_level_values("pair_id").str.split("__").str[0]
    per = delta.groupby(candidate).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), size=(n_boot, len(per)))
    draws = 100 * per["sum"].to_numpy()[idx].sum(1) / per["count"].to_numpy()[idx].sum(1)
    return tuple(float(v) for v in np.quantile(draws, [0.025, 0.975]))


def behavioral_checks(input_root: Path, out: Path, opc, n_boot: int, seed: int) -> dict:
    ref, ref_meta = load_scores(input_root, REF, opc)
    ref_index = pd.MultiIndex.from_frame(ref[opc.KEY])
    count = int((opc.snap(ref["margin"]) > 0).sum())
    ref_flags = opc._set_flags(ref, opc.snap(ref["margin"]) > 0).astype(float)
    border = ref[ref.condition == "attr_free"].groupby("pair_id").margin.mean().abs()
    borderline = set(border.index[border < 2.0])
    rows, overlaps, cuts, group_rows = [], [], [], []
    provenance = {REF: digest(score_paths(input_root, REF)[0])}
    for recipe, runs in RECIPES.items():
        flags_by_seed = {}
        for training_seed, run in zip((42, 43, 44), runs, strict=True):
            frame, meta = load_scores(input_root, run, opc)
            opc.check_comparable(ref_meta, meta)
            index = pd.MultiIndex.from_frame(frame[opc.KEY])
            if len(index) != len(ref_index) or len(index.difference(ref_index)):
                raise ValueError(f"{run}: prompt keys differ from SFT reference")
            flags, cut = opc.matched_flags(frame, "margin", count)
            flags = flags.reindex(ref_flags.index)
            if flags.isna().any():
                raise ValueError(f"{run}: counterfactual set keys differ")
            flags_by_seed[training_seed] = flags
            delta = flags - ref_flags
            lo, hi = paired_bootstrap(delta, n_boot, seed)
            on_border = flags.index.get_level_values("pair_id").isin(borderline)
            rows.append({"recipe": recipe, "seed": training_seed, "run": run,
                         "common_hire_rate_pct": 100 * count / len(ref),
                         "unstable_pct": 100 * flags.mean(),
                         "sft_unstable_pct": 100 * ref_flags.mean(),
                         "delta_pp": 100 * delta.mean(), "ci_low": lo, "ci_high": hi,
                         "borderline_pairs": len(borderline),
                         "on_borderline_share": float(flags[on_border].sum() / flags.sum()),
                         "sets": len(flags)})
            cuts.append({"recipe": recipe, "seed": training_seed, **cut})
            provenance[run] = digest(score_paths(input_root, run)[0])
        for (s, a), (t, b) in itertools.combinations(flags_by_seed.items(), 2):
            both, union, jac = weighted_jaccard(a, b)
            overlaps.append({"recipe": recipe, "seeds": f"{s}/{t}",
                             "common_hire_rate_pct": 100 * count / len(ref),
                             "unstable_a": a.sum(), "unstable_b": b.sum(),
                             "unstable_both_weighted": both, "unstable_union_weighted": union,
                             "weighted_jaccard": jac,
                             "sft_same_threshold_jaccard": 1.0 if ref_flags.sum() else np.nan})
        # Global cuts are fixed at the common hire rate before splitting into groups.
        for label, f in [*flags_by_seed.items(),
                         ("mean_42_43_44", pd.concat(flags_by_seed.values(), axis=1).mean(axis=1))]:
            for (family, cue), arm in f.groupby(level=["protected_group", "condition"]):
                reference = ref_flags.reindex(arm.index)
                d = arm - reference
                lo, hi = paired_bootstrap(d, n_boot, seed)
                group_rows.append({"recipe": recipe, "seed": label, "family": family,
                                   "cue_style": cue, "sets": len(arm),
                                   "candidates": arm.index.get_level_values("pair_id")
                                       .str.split("__").str[0].nunique(),
                                   "unstable_pct": 100 * arm.mean(),
                                   "sft_unstable_pct": 100 * reference.mean(),
                                   "delta_pp": 100 * d.mean(), "ci_low": lo, "ci_high": hi,
                                   "scope": "global cuts; candidate bootstrap conditional on evaluated seeds"})
    pd.DataFrame(rows).to_csv(out / "common_rate_seeds.csv", index=False)
    pd.DataFrame(overlaps).to_csv(out / "common_rate_overlap.csv", index=False)
    pd.DataFrame(cuts).to_csv(out / "common_rate_cuts.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out / "family_cue.csv", index=False)
    return {"status": "complete", "prompts": len(ref), "sets": len(ref_flags),
            "candidates": ref_flags.index.get_level_values("pair_id").str.split("__").str[0].nunique(),
            "common_hire_count": count, "common_hire_rate_pct": 100 * count / len(ref),
            "borderline_pairs": len(borderline), "score_sha256": provenance,
            "interval": "paired candidate percentile bootstrap, fixed global cuts",
            "jaccard": "sum(min(set weights)) / sum(max(set weights)); descriptive fuzzy membership overlap"}


def real_head_checks(root: Path, out: Path) -> dict:
    records, targets, absent = [], [], []
    # Base-selected rows are retained separately; no pooling across selection schemes.
    # Original Base-selected three-contrast study used the recovered SFT-v3 adapter
    # named `sft`. Later studies use its trained adapter named `sft_v3`.
    # Keep these actual spec names in the outputs rather than silently aliasing them.
    specs = ("base", "sft", "sft_v3", "dpo4880", "grpo_r4", "distill43")
    for spec in specs:
        for contrast in (CONTRASTS[:3] if spec == "sft" else CONTRASTS):
            selections = ("base",) if spec in ("base", "sft") else ("sft_v3",)
            if spec == "sft_v3" and contrast == "gen_male_female_impl":
                selections = ("base", "sft_v3")
            for select_on in selections:
                suffix = "" if select_on == "base" else "-sel_sft_v3"
                directory = root / "reports/interp/patching" / (
                    f"{spec}-grid-{contrast}-n64-s42{suffix}")
                paths = {k: directory / f"{k}.csv" for k in ("verify", "atp", "pairs")}
                if not all(p.exists() for p in paths.values()):
                    absent.append(str(directory.relative_to(root)))
                    continue
                v, a = (pd.read_csv(paths[k]) for k in ("verify", "atp"))
                v = v[v.cls == "suffix"].copy()
                a = a[(a.cls == "suffix") & (a.kind == "head_in")].copy()
                if v.duplicated(["layer", "head"]).any() or a.duplicated(["layer", "head"]).any():
                    raise ValueError(f"{directory}: duplicated suffix head summaries")
                v["is_control"] = v["control"].astype(str).str.lower().eq("true")
                v["real_rank_tested"] = v.removed_logits_per_pair.rank(ascending=False, method="min")
                a["atp_rank_all"] = a.removed_logits_per_pair.rank(ascending=False, method="min")
                joined = v.merge(a[["layer", "head", "removed_logits_per_pair", "atp_rank_all"]],
                                 on=["layer", "head"], suffixes=("_real", "_atp"), validate="one_to_one")
                rho = joined.removed_logits_per_pair_real.rank().corr(
                    joined.removed_logits_per_pair_atp.rank())
                ordered = joined.sort_values("removed_logits_per_pair_real", ascending=False)
                pairs = pd.read_csv(paths["pairs"])
                identity = {"spec": spec, "contrast": contrast, "select_on": select_on,
                            "pairs": len(pairs), "verified_heads": len(v), "all_atp_heads": len(a)}
                records.append({**identity, "spearman_verified_subset": rho,
                                "top_real_verified": " ".join(
                                    f"L{int(r.layer)}.H{int(r.head)}" for r in ordered.head(3).itertuples()),
                                "max_abs_random_share": v.loc[v.is_control, "share_removed"].abs().max(),
                                "verify_sha256": digest(paths["verify"]),
                                "scope": "suffix positions; ranks only among tested heads, not all heads"})
                for layer, head in TARGET_HEADS:
                    hit = joined[(joined.layer == layer) & (joined["head"] == head)]
                    atp_hit = a[(a.layer == layer) & (a["head"] == head)]
                    target = {**identity, "target": f"L{layer}.H{head}", "verified": not hit.empty,
                              "atp_rank_all": int(atp_hit.atp_rank_all.iloc[0]) if len(atp_hit) else np.nan}
                    if len(hit):
                        row = hit.iloc[0]
                        target.update(real_rank_tested=int(row.real_rank_tested),
                                      real_removed_logits_per_pair=float(row.removed_logits_per_pair_real),
                                      atp_removed_logits_per_pair=float(row.removed_logits_per_pair_atp),
                                      real_share_removed=float(row.share_removed),
                                      is_random_control=bool(row.is_control))
                        for key in ("ci_low", "ci_high", "share_ci_low", "share_ci_high"):
                            if key in row:
                                target["real_" + key] = row[key]
                    targets.append(target)
    if not records:
        raise FileNotFoundError("no complete {verify,atp,pairs}.csv intervention summaries")
    pd.DataFrame(records).to_csv(out / "head_verification.csv", index=False)
    pd.DataFrame(targets).to_csv(out / "target_heads.csv", index=False)
    return {"status": "complete" if not absent else "partial", "runs_summarized": len(records),
            "missing_runs": absent,
            "caution": "AtP selected most tested heads; subset correlation/rank does not validate a full circuit or a global real-head rank"}


def full_grid_effects(input_root: Path, out: Path, n_boot: int, seed: int) -> dict:
    path = input_root / "outputs/interp/manifest/contrasts-grid-en.parquet"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    frame = pd.read_parquet(path)
    frame = frame[frame.aligned & frame.contrast_id.isin(CONTRASTS)].copy()
    rows = []
    for contrast, group in frame.groupby("contrast_id", sort=False):
        base, sft = group.dm_base.abs(), group.dm_sft_v3.abs()
        per = group.assign(abs_base=base, abs_sft=sft).groupby("candidate_id")[["abs_base", "abs_sft"]].sum()
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(per), size=(n_boot, len(per)))
        denominator = per.abs_base.to_numpy()[idx].sum(1)
        draws = 100 * (1 - per.abs_sft.to_numpy()[idx].sum(1) / denominator)
        low, high = np.quantile(draws, [0.025, 0.975])
        rows.append({"contrast": contrast, "pairs": len(group), "candidates": len(per),
                     "base_mean_abs_effect": base.mean(), "sft_mean_abs_effect": sft.mean(),
                     "removed_pct": 100 * (1 - sft.sum() / base.sum()),
                     "ci_low": low, "ci_high": high,
                     "scope": "all aligned pairs, not effect-selected patching subset"})
    result = pd.DataFrame(rows)
    if set(result.contrast) != set(CONTRASTS):
        raise ValueError("manifest lacks one or more poster contrasts")
    result.to_csv(out / "full_grid_effects.csv", index=False)
    tails = []
    for method, spec in SPECS.items():
        column = "dm_" + spec
        if column not in frame:
            continue
        for contrast, group in frame.groupby("contrast_id", sort=False):
            valid = group[["dm_sft_v3", column]].dropna()
            ref, arm = valid.dm_sft_v3.abs(), valid[column].abs()
            tails.append({"method": method, "spec": spec, "contrast": contrast, "pairs": len(valid),
                          "p50_ratio": arm.median() / ref.median(),
                          "p99_ratio": arm.quantile(0.99) / ref.quantile(0.99),
                          "mean_abs_ratio": arm.mean() / ref.mean(),
                          "scope": "one selected checkpoint; quantiles over aligned pairs, descriptive"})
    if tails:
        pd.DataFrame(tails).to_csv(out / "tail_sensitivity.csv", index=False)
    return {"status": "complete", "source_sha256": digest(path),
            "contrasts": result.contrast.tolist(), "tail_rows": len(tails)}


def existing_tail_tables(root: Path, out: Path) -> dict:
    rows, missing = [], []
    for method, spec in SPECS.items():
        path = root / "reports/interp/distill" / spec / "sensitivity.csv"
        if not path.exists():
            missing.append(str(path.relative_to(root)))
            continue
        frame = pd.read_csv(path)
        for row in frame.itertuples(index=False):
            rows.append({"method": method, "spec": spec, "contrast": row.contrast,
                         "pairs": row.pairs, "p50_ratio": row.r_attr_median,
                         "p99_ratio": row.r_attr_p99, "mean_abs_ratio": row.r_attr,
                         "margin_scale_ratio": row.r_scale,
                         "scope": "one selected checkpoint; all aligned pairs; descriptive quantiles"})
    if not rows:
        raise FileNotFoundError("no existing sensitivity.csv tables")
    pd.DataFrame(rows).to_csv(out / "tail_sensitivity_from_reports.csv", index=False)
    return {"status": "complete" if not missing else "partial", "rows": len(rows), "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-root", type=Path, default=Path(os.environ.get(
        "HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw")))
    ap.add_argument("--out", type=Path, default=ROOT / "reports/poster_followup")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--inventory", action="store_true")
    args = ap.parse_args()
    if args.inventory:
        paths = [str(p) for run in (REF, *itertools.chain.from_iterable(RECIPES.values()))
                 for p in score_paths(args.input_root, run)]
        paths.append(str(args.input_root / "outputs/interp/manifest/contrasts-grid-en.parquet"))
        print("\n".join(paths))
        return 0
    if args.n_boot < 100:
        raise SystemExit("use at least 100 bootstrap draws")
    args.out.mkdir(parents=True, exist_ok=True)
    opc = load_matching()
    result = {"analysis": "poster CPU follow-up", "canonical_matching_sha256": opc.source_sha256,
              "n_boot": args.n_boot, "seed": args.seed,
              "scope": "exploratory development benchmark; no new model inference",
              "checks": {}}
    checks = {
        "behavioral": lambda: behavioral_checks(args.input_root, args.out, opc, args.n_boot, args.seed),
        "real_heads": lambda: real_head_checks(ROOT, args.out),
        "full_grid_effects": lambda: full_grid_effects(args.input_root, args.out, args.n_boot, args.seed),
        "existing_tail_tables": lambda: existing_tail_tables(ROOT, args.out),
    }
    for name, check in checks.items():
        try:
            result["checks"][name] = check()
        except (FileNotFoundError, ImportError) as error:
            result["checks"][name] = {"status": "pending", "reason": str(error)}
    required = ("behavioral", "real_heads", "full_grid_effects")
    result["complete"] = all(result["checks"][name]["status"] == "complete" for name in required)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["complete"] or args.allow_missing else 2


if __name__ == "__main__":
    raise SystemExit(main())
