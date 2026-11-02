"""CPU analysis of completed final-adapter merge-precision inference.

Each HF representation/readout is calibrated on its own scores. The only value
borrowed from the canonical vLLM reference is its target hire COUNT, not a
numerical threshold. Intervals resample candidates with all their rows retained,
holding the originally calibrated cuts fixed. Three-seed means are descriptive.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import poster_audit_followup as P

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("full_r32", "window_r32", "window_r163")
SEEDS = (42, 43, 44)
MODES = ("unmerged", "merged_bf16")
READOUTS = ("margin_fp32", "margin_bf16")


def check_alignment(a, b, keys):
    if a[keys].isna().any().any() or a.duplicated(keys).any():
        raise ValueError("missing or duplicated prompt key")
    if b[keys].isna().any().any() or b.duplicated(keys).any():
        raise ValueError("missing or duplicated prompt key")
    if not np.array_equal(a[keys].to_numpy(dtype=object), b[keys].to_numpy(dtype=object)):
        raise ValueError("score rows do not align")


def flags_at_count(frame, column, count, opc):
    """Whole-cut tie mixing; fp32 readout is not rounded to the vLLM grid."""
    m = np.asarray(frame[column], dtype=float)
    if not np.isfinite(m).all() or not 0 <= count <= len(m):
        raise ValueError("invalid scores or target hire count")
    if column == "margin_bf16":
        m = opc.snap(m)
    level, weight = opc.cut_for_count(m, count)
    strict = opc._set_flags(frame, m > level).astype(float)
    inclusive = opc._set_flags(frame, m >= level).astype(float)
    flags = (1 - weight) * strict + weight * inclusive
    achieved = int((m > level).sum()) + weight * int((m == level).sum())
    if not np.isclose(achieved, count) or not 0 <= weight <= 1:
        raise ValueError("cut does not match target count")
    return flags, {"threshold": level, "tie_weight": weight,
                   "tied_prompts": int((m == level).sum()), "hire_count": count,
                   "hire_rate_pct": 100 * count / len(m)}


def interval(delta, n_boot, seed):
    if delta.isna().any():
        raise ValueError("unaligned or nonfinite set differences")
    low, high = P.paired_bootstrap(delta, n_boot, seed)
    return {"delta_pp": 100 * float(delta.mean()), "ci_low_pp": low, "ci_high_pp": high}


def difference_in_differences(window_unmerged, full_unmerged, window_merged, full_merged):
    all_flags = pd.concat([window_unmerged, full_unmerged, window_merged, full_merged], axis=1)
    if all_flags.isna().any().any():
        raise ValueError("unaligned flags in difference-in-differences")
    return (all_flags.iloc[:, 0] - all_flags.iloc[:, 1]) - (all_flags.iloc[:, 2] - all_flags.iloc[:, 3])


def load_completed(directory, opc):
    summary = json.loads((directory / "summary.json").read_text())
    identity = json.loads((directory / "identity.json").read_text())
    inputs = json.loads((directory / "inputs.json").read_text())
    if summary.get("status") != "complete" or summary.get("mode") != "full":
        raise ValueError(f"not a complete full pass: {directory}")
    for field in ("common_sft_weight_parity", "served_weight_parity"):
        if not summary[field].get("all_unique_parameters_bit_identical"):
            raise ValueError(f"weight-parity gate failed: {directory}")
    result, hashes = {}, {}
    for mode in MODES:
        paths = sorted((directory / mode).glob("part-*.parquet"))
        if not paths or [p.name for p in paths] != [f"part-{i:05d}.parquet" for i in range(len(paths))]:
            raise ValueError(f"missing score chunks: {directory}/{mode}")
        parts = []
        for p in paths:
            part = pd.read_parquet(p)
            expected = summary["score_artifacts"][str(p.relative_to(directory))]
            if P.digest(p) != expected["sha256"] or len(part) != expected["rows"]:
                raise ValueError(f"completed score artifact changed: {p}")
            parts.append(part)
        f = pd.concat(parts, ignore_index=True)
        if len(f) != 31050 or f.prompt_uid.tolist() != inputs["prompt_uids"]:
            raise ValueError(f"incomplete full score pass: {directory}/{mode}")
        if f.prompt_uid.duplicated().any() or not np.isfinite(f[list(READOUTS)].to_numpy()).all():
            raise ValueError(f"invalid scores: {directory}/{mode}")
        if f.pair_id.nunique() != 450 or f.candidate_id.nunique() != 150:
            raise ValueError(f"unexpected benchmark population: {directory}/{mode}")
        f = f.sort_values(opc.KEY).reset_index(drop=True)
        check_alignment(f, f, opc.KEY)
        result[mode] = f
        for p in paths:
            hashes[str(p)] = P.digest(p)
    check_alignment(result[MODES[0]], result[MODES[1]], opc.KEY + ["prompt_uid"])
    for name in ("summary.json", "identity.json", "inputs.json", "execution.json"):
        p = directory / name
        hashes[str(p)] = P.digest(p)
    return result, identity, inputs, hashes


def analyze(source, output, input_root, n_boot=1000, bootstrap_seed=42):
    opc = P.load_matching()
    canonical, _ = P.load_scores(input_root, P.REF, opc)
    count = int((opc.snap(canonical.margin) > 0).sum())
    frames, sources, common_signature, full_input_signature = {}, {}, None, None
    for seed in SEEDS:
        for arm in ARMS:
            directory = source / "full" / f"{arm}_seed{seed}"
            loaded, identity, inputs, hashes = load_completed(directory, opc)
            if identity["seed"] != seed or identity["arm"] != arm:
                raise ValueError("mislabeled experiment directory")
            signature = identity["common"]
            input_signature = {k: inputs[k] for k in ("full_grid_sha256", "selected_sha256", "tokenization_sha256")}
            if common_signature is not None and (signature != common_signature or input_signature != full_input_signature):
                raise ValueError("experiment scoring inputs/settings differ across arms")
            common_signature, full_input_signature = signature, input_signature
            frames[(seed, arm)] = loaded
            sources.update(hashes)
    base = frames[(42, "full_r32")]["merged_bf16"]
    check_alignment(canonical.sort_values(opc.KEY).reset_index(drop=True), base, opc.KEY)
    for modes in frames.values():
        for f in modes.values():
            check_alignment(base, f, opc.KEY + ["prompt_uid"])
    primary, within, own, did, curves, cuts = ([] for _ in range(6))
    flags = {}
    for (seed, arm), modes in frames.items():
        for mode, f in modes.items():
            for column in READOUTS:
                ident = {"seed": seed, "arm": arm, "representation": mode, "readout": column}
                common, cut = flags_at_count(f, column, count, opc)
                flags[(seed, arm, mode, column)] = common
                cuts.append({**ident, "basis": "common_hire_count", **cut})
                own_flags = opc._set_flags(f, f[column].to_numpy() > 0).astype(float)
                own.append({**ident, "hire_rate_pct": 100 * float((f[column] > 0).mean()),
                            "unstable_pct": 100 * float(own_flags.mean()),
                            "common_rate_unstable_pct": 100 * float(common.mean())})
                for rate in np.linspace(10, 35, 51):
                    k = round(float(rate) * len(f) / 100)
                    curve_flags, _ = flags_at_count(f, column, k, opc)
                    curves.append({**ident, "target_hire_rate_pct": rate,
                                   "actual_hire_rate_pct": 100 * k / len(f),
                                   "unstable_pct": 100 * float(curve_flags.mean())})
        for column in READOUTS:
            u, m = modes["unmerged"][column], modes["merged_bf16"][column]
            delta = flags[(seed, arm, "unmerged", column)] - flags[(seed, arm, "merged_bf16", column)]
            within.append({"seed": seed, "arm": arm, "readout": column,
                           "comparison": "unmerged_minus_merged_at_common_rate", **interval(delta, n_boot, bootstrap_seed),
                           "mean_absolute_margin_change": float((u - m).abs().mean()),
                           "p99_absolute_margin_change": float((u - m).abs().quantile(.99)),
                           "greedy_decision_disagreement_pct": 100 * float(((u > 0) != (m > 0)).mean())})
    for seed in SEEDS:
        for column in READOUTS:
            for arm in ARMS[1:]:
                for mode in MODES:
                    full = frames[(seed, "full_r32")][mode]
                    window = frames[(seed, arm)][mode]
                    own_count = int((full[column] > 0).sum())
                    for basis, k in (("common_hire_count", count), ("same_representation_full_own_hire_count", own_count)):
                        f, full_cut = flags_at_count(full, column, k, opc)
                        w, window_cut = flags_at_count(window, column, k, opc)
                        primary.append({"seed": seed, "window": arm, "representation": mode,
                                        "readout": column, "basis": basis, "hire_rate_pct": 100 * k / len(full),
                                        "window_unstable_pct": 100 * float(w.mean()),
                                        "full_unstable_pct": 100 * float(f.mean()),
                                        **interval(w - f, n_boot, bootstrap_seed)})
                delta = difference_in_differences(
                    flags[(seed, arm, "unmerged", column)], flags[(seed, "full_r32", "unmerged", column)],
                    flags[(seed, arm, "merged_bf16", column)], flags[(seed, "full_r32", "merged_bf16", column)])
                did.append({"seed": seed, "window": arm, "readout": column,
                            "comparison": "window_minus_full_gap_unmerged_minus_merged",
                            **interval(delta, n_boot, bootstrap_seed)})
    tables = {"window_comparisons": primary, "within_checkpoint": within,
              "own_rates": own, "gap_change": did, "curves": curves, "cuts": cuts}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
    mean = pd.DataFrame(primary).groupby(["window", "representation", "readout", "basis"]).delta_pp.agg(["mean", "min", "max"]).reset_index()
    mean.to_csv(output / "descriptive_seed_means.csv", index=False)
    reference_path = P.score_paths(input_root, P.REF)[0]
    result = {"complete": True, "seeds": list(SEEDS), "arms": list(ARMS),
              "common_hire_count": count, "common_hire_rate_pct": 100 * count / len(base),
              "reference_hire_count_source": str(reference_path), "reference_sha256": P.digest(reference_path),
              "source_sha256": sources, "canonical_matching_sha256": opc.source_sha256,
              "scoring_common": common_signature, "inputs": full_input_signature,
              "n_boot": n_boot, "bootstrap_seed": bootstrap_seed,
              "interval_scope": "paired candidate bootstrap with fixed cuts, conditional on evaluated checkpoints",
              "seed_means_scope": "descriptive only; no aggregate three-seed interval",
              "positive_gap_change": "unmerged representation increases window-minus-full instability gap",
              "backend_scope": "same HF backend; fp32 readout unsnapped, native bf16 readout uses canonical snap",
              "code_sha256": {str(Path(__file__).relative_to(ROOT)): P.digest(Path(__file__)),
                              "scripts/poster_audit_followup.py": P.digest(ROOT / "scripts/poster_audit_followup.py")},
              "output_sha256": {p.name: P.digest(p) for p in sorted(output.glob("*.csv"))}}
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "summary.json")
    (output / "pending.json").unlink(missing_ok=True)
    print(json.dumps({"complete": True, "output": str(output), "tables": {k: len(v) for k, v in tables.items()}}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "reports/research_followup/merge_precision")
    parser.add_argument("--out", type=Path, default=ROOT / "reports/research_followup/merge_precision/aggregate")
    parser.add_argument("--input-root", type=Path, default=Path(os.environ.get("HBM_OUTPUT_ROOT", ROOT)))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--when-ready", action="store_true", help="Called by each full Slurm job; aggregate only after all nine arms finish")
    args = parser.parse_args()
    if args.n_boot < 100:
        parser.error("use at least 100 bootstrap draws")
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / ".aggregate.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if not args.when_ready:
            raise
        print(json.dumps({"complete": False, "status": "another_aggregator_running"}))
        raise SystemExit(0)
    expected = [args.source / "full" / f"{arm}_seed{seed}" / "summary.json" for seed in SEEDS for arm in ARMS]
    missing = [str(p.parent.relative_to(args.source)) for p in expected if not p.exists()]
    if args.when_ready and missing:
        pending = {"complete": False, "status": "waiting_for_full_arms", "missing": missing}
        (args.out / "pending.json").write_text(json.dumps(pending, indent=2) + "\n")
        print(json.dumps(pending))
    else:
        analyze(args.source, args.out, args.input_root, args.n_boot)
