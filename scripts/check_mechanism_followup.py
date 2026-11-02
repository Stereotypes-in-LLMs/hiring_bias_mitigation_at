"""Validate and reuse completed mechanism artifacts for the frozen extra seeds.

Exit 3 means incomplete, 1 means invalid. This distinction lets the GPU runner
resume missing work while refusing a completed pass that fails its audit check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hiring_bias_mitigation.interp.io import interp_dir, report_dir
from hiring_bias_mitigation.interp.spec import load_spec_set, resolve_adapter, adapter_fingerprint
from hiring_bias_mitigation.utils.config import resolve_output_path

SPECS = ("dpo4880_s43", "dpo4880_s44", "grpo_r4_s43", "grpo_r4_s44", "distill42", "distill44")
CONTRASTS = ("mil_combat_civ_impl", "mil_combat_civ_expl", "rel_atheist_christian_impl", "gen_male_female_impl")
SUFFIX = "-n64-s42-sel_sft_v3"


def fingerprints(spec: str) -> dict:
    specs = load_spec_set()
    return {a: adapter_fingerprint(resolve_adapter(specs.adapter_paths[a]))
            for a in specs.specs[spec].adapters}


def cache_identity(directory: Path, current: dict, *, prepare=False,
                   legacy_allowed=False, native_meta: Path | None = None) -> dict:
    """Bind a new cache before GPU work; never silently adopt ambiguous partials."""
    identity = directory / "poster_followup.meta.json"
    if identity.exists():
        saved = json.loads(identity.read_text())
        if saved.get("adapters") != current:
            raise ValueError(f"adapter identity changed for cache: {directory}")
        return saved
    native = json.loads(native_meta.read_text()) if native_meta and native_meta.exists() else {}
    if "adapters" in native and native["adapters"] != current:
        raise ValueError(f"native adapter fingerprint differs for cache: {directory}")
    occupied = any(path.is_file() for path in directory.rglob("*"))
    verified_native = native.get("adapters") == current
    if occupied and not verified_native and not legacy_allowed:
        raise ValueError(f"ambiguous existing cache has no adapter fingerprint: {directory}")
    provenance = ("existing_cache_native_fingerprint_matches" if verified_native else
                  "legacy_cache_adapter_identity_unverified" if occupied else
                  "empty_cache_bound_before_first_compute")
    record = {"adapters": current, "prior_cache_provenance": provenance}
    if prepare:
        directory.mkdir(parents=True, exist_ok=True)
        identity.write_text(json.dumps(record, indent=2) + "\n")
    return record


def cache_checks(spec: str, *, prepare=False) -> None:
    current = fingerprints(spec)
    directory = interp_dir("margins", spec)
    cache_identity(directory, current, prepare=prepare, legacy_allowed=spec == "distill42",
                   native_meta=directory / "grid-en.parquet.meta.json")
    for contrast in CONTRASTS:
        directory = interp_dir("patching", spec, "grid", f"{contrast}{SUFFIX}")
        cache_identity(directory, current, prepare=prepare, legacy_allowed=spec == "distill42")


def incomplete(message: str) -> None:
    print(message)
    raise SystemExit(3)


def check_margins(spec: str) -> None:
    path = interp_dir("margins", spec) / "grid-en.parquet"
    summary = report_dir() / f"margins_summary-{spec}-grid-en.json"
    meta = Path(str(path) + ".meta.json")
    current = fingerprints(spec)
    cache_identity(path.parent, current, legacy_allowed=spec == "distill42", native_meta=meta)
    if not all(p.exists() for p in (path, summary, meta)):
        incomplete(f"margin artifacts incomplete for {spec}")
    recorded = json.loads(summary.read_text())
    audit = recorded.get("audit", {})
    if (recorded.get("rows") != 31050 or not audit.get("available")
            or audit.get("rows_matched") != 31050 or audit.get("agreement_pct", 0) < 97):
        raise ValueError(f"completed margin pass fails the full-grid audit check: {spec}: {audit}")
    frame = pd.read_parquet(path, columns=["prompt_uid"])
    if len(frame) != 31050 or frame["prompt_uid"].nunique() != 31050:
        raise ValueError(f"margin frame has missing or duplicate prompts: {path}")
    if json.loads(meta.read_text()).get("adapters") != current:
        raise ValueError(f"adapter fingerprint changed since margin pass: {spec}")
    print(f"margin artifacts ready: {spec}, agreement {audit['agreement_pct']:.3f}%")


def check_patch(spec: str, contrast: str) -> None:
    name = f"{spec}-grid-{contrast}{SUFFIX}"
    rep = report_dir("patching", name)
    directory = interp_dir("patching", spec, "grid", f"{contrast}{SUFFIX}")
    cache_identity(directory, fingerprints(spec), legacy_allowed=spec in ("distill42", "sft_v3"))
    needed = [rep / f"{part}.csv" for part in ("pairs", "real", "atp", "verify")]
    if not all(p.exists() and p.stat().st_size for p in needed):
        incomplete(f"patch artifacts incomplete: {name}")
    if spec not in ("distill42", "sft_v3") and not (directory / "poster_followup.meta.json").exists():
        raise ValueError(f"completed patch summaries lack adapter identity: {name}")
    ref = report_dir("patching", f"sft_v3-grid-{contrast}{SUFFIX}") / "pairs.csv"
    if not ref.exists():
        raise ValueError(f"reference-selected pair list missing: {ref}")
    actual = pd.read_csv(rep / "pairs.csv")
    expected = pd.read_csv(ref)
    if set(actual["pair_uid"]) != set(expected["pair_uid"]) or len(actual) != len(expected):
        raise ValueError(f"patch pair selection differs from SFT-v3 reference: {name}")
    if len(expected) != (45 if contrast == "gen_male_female_impl" else 64):
        raise ValueError(f"reference selection size changed: {contrast}, {len(expected)}")
    for part in ("real", "atp", "verify"):
        if pd.read_csv(rep / f"{part}.csv").empty:
            raise ValueError(f"empty {part} summary: {name}")
    print(f"patch artifacts ready: {name}, {len(actual)} identical reference-selected pairs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, choices=SPECS)
    parser.add_argument("--check", choices=("inputs", "margins", "patch", "prepare-caches"), required=True)
    parser.add_argument("--contrast", choices=CONTRASTS)
    args = parser.parse_args()
    if args.check == "prepare-caches":
        cache_checks(args.spec)  # Validate every cache before recording any new identity.
        cache_checks(args.spec, prepare=True)
    elif args.check == "margins":
        check_margins(args.spec)
    elif args.check == "patch":
        if args.contrast is None:
            parser.error("--contrast is required for --check patch")
        check_patch(args.spec, args.contrast)
    else:
        specs = load_spec_set()
        for adapter in specs.specs[args.spec].adapters:
            path = Path(resolve_adapter(specs.adapter_paths[adapter]))
            if not (path / "adapter_model.safetensors").exists():
                raise ValueError(f"adapter file missing: {path}")
        for required in (
            interp_dir("manifest") / "contrasts-grid-en.parquet",
            interp_dir("margins", "sft_v3") / "grid-en.parquet",
        ):
            if not required.exists():
                raise ValueError(f"shared input missing: {required}")
        # Import the script's established mapping without running the margin pass.
        import runpy
        audit_run = runpy.run_path(str(ROOT / "scripts/interp/02_margins.py"))["AUDIT_RUN"][args.spec]
        audit_path = Path(resolve_output_path("outputs/raw")) / f"{audit_run}.parquet"
        if not audit_path.exists():
            raise ValueError(f"behavioural audit missing: {audit_path}")
        for contrast in CONTRASTS:
            check_patch("sft_v3", contrast)
        cache_checks(args.spec)
        print(f"inputs ready for frozen model {args.spec}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
