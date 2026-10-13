"""Generates v2 negatives -- covert, decision-flipped `rejected` responses -- on v1's prompts.

    python scripts/generate_negatives_v2.py --config configs/generation/negatives_v2_small.yaml
    python scripts/generate_negatives_v2.py --config <cfg> --dry-run     # prompts only, no model
    python scripts/generate_negatives_v2.py --config <cfg> --from-hub      # fetch v1 raw first
    python scripts/generate_negatives_v2.py --config <cfg> --strategies covert_flip

Reuses v1's prompt, reference and chosen side unchanged (`raw/<lang>_invariant.parquet`, with
the chosen side reproduced by v1's own filter) and generates only the `rejected` side, three
ways, then judges and filters it. Every teacher pass is dumped under `raw/` and resumed from
there, so an interrupted run continues. `--dry-run` writes the batch and every prompt and
loads no model. See `generation/negatives_v2.py` for the construction and
docs/superpowers/specs/2026-10-13-dpo-negatives-v2-design.md for the design.

After it: `python scripts/check_preference_direction.py --batch <artifacts_dir>`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.eval import backends as BK  # noqa: E402
from hiring_bias_mitigation.generation import negatives_v2 as N  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    load_config,
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402
from hiring_bias_mitigation.utils.seed import set_seed  # noqa: E402

log = get_logger("generate_negatives_v2")

V1_VALIDATION_FILES = ("sft_validation.parquet", "dpo_validation.parquet")


def fetch_v1_from_hub(repo: str, source_dir: Path, lang: str) -> None:
    """The public copy of v1's raw passes and validation splits, into the v1 artifacts dir."""
    from huggingface_hub import hf_hub_download

    wanted = [f"data/raw/{lang}_invariant.parquet", *(f"data/{f}" for f in V1_VALIDATION_FILES)]
    for name in wanted:
        target = source_dir / name.split("data/", 1)[1]
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        path = hf_hub_download(repo, name, repo_type="dataset")
        target.write_bytes(Path(path).read_bytes())
        log.info("fetched %s -> %s", name, target)


def v1_validation_candidates(source_dir: Path) -> set[str]:
    """Every candidate v1 held out, so v2's split keeps them held out too."""
    ids: set[str] = set()
    for name in V1_VALIDATION_FILES:
        path = source_dir / name
        if path.exists():
            ids |= set(pd.read_parquet(path, columns=["candidate_id"])["candidate_id"].astype(str))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="sample the batch and write every prompt; load no model")
    parser.add_argument("--from-hub", action="store_true",
                        help="fetch v1's raw passes from the public dataset if they are missing")
    parser.add_argument("--strategies", nargs="+", choices=N.STRATEGIES,
                        help="override the config's strategy list")
    parser.add_argument("--rows-per-cell", type=int, help="override the config's batch size")
    parser.add_argument("--artifacts-dir", help="override the config's output directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    if args.strategies:
        cfg["strategies"] = args.strategies
    if args.rows_per_cell:
        cfg["rows_per_cell"] = args.rows_per_cell
    lang = str(cfg.get("lang", "en"))

    require_output_root()
    source_dir = Path(resolve_output_path(cfg["source_artifacts_dir"]))
    out_dir = Path(resolve_output_path(args.artifacts_dir or cfg["artifacts_dir"]))
    invariant_path = source_dir / "raw" / f"{lang}_invariant.parquet"
    if args.from_hub:
        fetch_v1_from_hub(cfg["source_hub_repo"], source_dir, lang)
    if not invariant_path.exists():
        raise SystemExit(
            f"{invariant_path} is missing. Point source_artifacts_dir at the v1 artifacts, or "
            "pass --from-hub to fetch the public copy."
        )
    invariant_raw = pd.read_parquet(invariant_path)
    log.info("v1 invariant pass: %d rows from %s", len(invariant_raw), invariant_path)
    v1_val_ids = v1_validation_candidates(source_dir)
    source_label = f"{cfg['source_artifacts_dir']} ({invariant_path.name})"

    if args.dry_run:
        report = N.run_batch(cfg, invariant_raw, backend=None, out_dir=out_dir,
                             source_artifacts=source_label, dry_run=True, v1_val_ids=v1_val_ids)
        print(f"dry run: {report['batch']['rows']} prompts per strategy written to {out_dir}")
        for cell, n in report["batch"]["cells"].items():
            print(f"  {cell}: {n}")
        return

    backend = BK.build_backend(N.teacher_backend_config(cfg))
    try:
        report = N.run_batch(cfg, invariant_raw, backend, out_dir, source_label,
                             v1_val_ids=v1_val_ids)
    finally:
        backend.close()

    print(f"\nbatch written to {out_dir}")
    for strategy, entry in report["strategies"].items():
        covert = entry["covertness"]
        print(
            f"  {strategy:13s} kept {entry['n_kept']:4d}/{entry['n_prompts']} "
            f"(yield {entry['yield']:.0%}) | raw: refusals {covert['refusal_rate']:.0%}, "
            f"attribute named {covert['raw_attribute_mention_rate']:.0%}, "
            f"group named {covert['raw_group_mention_rate']:.0%} | "
            f"balanced AUC {entry['separability']['auc_balanced']}"
        )
    print(f"  splits: {report['splits']}")
    print(f"\nread {out_dir / 'samples.md'} before the direction check.")


if __name__ == "__main__":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    main()
