"""CPU dry run of the external part of the teacher's training pool, on the real staged data.

    python scripts/build_external_pool.py --config configs/generation/teacher_external_draft.yaml
    python scripts/build_external_pool.py --config <cfg> --rjdb-include-weakened true
    python scripts/build_external_pool.py --config <cfg> --out-dir <dir>

This builds exactly the external rows that `generate_training_data.py` would send to the
teacher for the config: the same `pool.pool_config`, the same counts and seed, and the same
holdout asserts. It does not download Djinni and it loads no model. Djinni is dropped from
`pool_sources` here, because its pool is V1's and is not what a dry run of the external
sources is for. It writes three files to `--out-dir` (default
`$HBM_EXTERNAL_ROOT/_pool_dryrun`, outside git):

    pool_<lang>.parquet               the pool rows
    funnel_<lang>.json                the rows left after every filter, per source, plus the
                                      teacher cost of this pool
    reference_prompts_<lang>.txt      3 rendered teacher reference prompts per source, for a
                                      human to read before any GPU time is spent

The staged data lives outside the repository, under `$HBM_EXTERNAL_ROOT` (default
`$HBM_OUTPUT_ROOT/external_data`). A missing table fails with the command that fetches it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data import external as X  # noqa: E402
from hiring_bias_mitigation.generation import pool as PL  # noqa: E402
from hiring_bias_mitigation.generation import synth as SY  # noqa: E402
from hiring_bias_mitigation.utils.config import load_config  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("build_external_pool")

PROMPTS_PER_SOURCE = 3


def teacher_cost(cfg: dict, n_pairs: int) -> dict:
    """Upper bound on teacher calls for a pool, with the same expansion as generate_training_data.

    Passes 2 and 3 run on every attribute variant of a pair whose reference survived its
    filter. So the bound assumes every reference survives.
    """
    groups = list(cfg.get("protected_groups", ("military_status", "gender", "religion")))
    conditions = list(cfg.get("conditions", ("explicit", "implicit")))
    variants = len(groups) * int(cfg.get("attributes_per_pair", 2)) * len(conditions)
    biased = cfg.get("rejected_source", "teacher_biased") in ("teacher_biased", "both")
    calls = {"reference": n_pairs, "invariant": n_pairs * variants,
             "biased": n_pairs * variants if biased else 0}
    return {"pairs": n_pairs, "variants_per_pair": variants, "calls_by_pass": calls,
            "calls_upper_bound": sum(calls.values())}


def render_prompts(pool, lang: str, seed: int) -> str:
    """The first teacher prompt each pair will get, for PROMPTS_PER_SOURCE pairs per source."""
    blocks = []
    for source, part in pool.groupby("source", sort=False):
        sample = part.sample(n=min(PROMPTS_PER_SOURCE, len(part)), random_state=seed)
        for (_, row), prompt in zip(sample.iterrows(), SY.reference_prompts(sample, lang)):
            blocks.append(
                f"{'=' * 100}\nsource={source}  pair_id={row['pair_id']}  "
                f"candidate_id={row['candidate_id']}  occupation={row['occupation']}  "
                f"cv_variant={row['cv_variant']}\n{'=' * 100}\n{prompt}\n"
            )
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/generation/teacher_external_draft.yaml")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--out-dir", help="defaults to $HBM_EXTERNAL_ROOT/_pool_dryrun")
    parser.add_argument("--rjdb-include-weakened", choices=("config", "true", "false"),
                        default="config", help="override the config's rjdb_include_weakened")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.rjdb_include_weakened != "config":
        cfg["rjdb_include_weakened"] = args.rjdb_include_weakened == "true"
    pool_cfg = PL.pool_config(cfg, args.lang)
    pool_cfg = replace(pool_cfg, sources=tuple(s for s in pool_cfg.sources if s != "djinni"))
    if not pool_cfg.sources:
        raise SystemExit(f"{args.config}: pool_sources names no external source")

    out_dir = Path(args.out_dir) if args.out_dir else X.external_root() / "_pool_dryrun"
    out_dir.mkdir(parents=True, exist_ok=True)
    pool, funnel = PL.build_pool_with_report(pool_cfg)
    if pool.empty:
        raise SystemExit(f"no external pairs for {args.lang}; see the log for which filter")

    pool_path = out_dir / f"pool_{args.lang}.parquet"
    pool.to_parquet(pool_path, index=False)
    report = {
        "config": str(args.config),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "external_root": str(X.external_root()),
        "lang": args.lang,
        "seed": pool_cfg.seed,
        "pool_sources": list(pool_cfg.sources),
        "pairs_per_source": pool_cfg.pairs_per_source,
        "jobs_per_candidate": pool_cfg.jobs_per_candidate,
        "rjdb_include_weakened": pool_cfg.rjdb_include_weakened,
        "rows": len(pool),
        "rows_by_source": {str(k): int(v) for k, v in pool["source"].value_counts().items()},
        "distinct_candidates_by_source": {
            str(k): int(v) for k, v in pool.groupby("source")["candidate_id"].nunique().items()
        },
        "cv_chars_median_by_source": {
            str(k): int(v) for k, v in pool.groupby("source")["cv"].agg(
                lambda s: s.str.len().median()).items()
        },
        "jd_chars_median_by_source": {
            str(k): int(v) for k, v in pool.groupby("source")["job_description"].agg(
                lambda s: s.str.len().median()).items()
        },
        "teacher_cost": teacher_cost(cfg, len(pool)),
        "funnel": funnel,
    }
    funnel_path = out_dir / f"funnel_{args.lang}.json"
    funnel_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    prompts_path = out_dir / f"reference_prompts_{args.lang}.txt"
    prompts_path.write_text(render_prompts(pool, args.lang, pool_cfg.seed), encoding="utf-8")

    print(f"\n{len(pool):,} external pairs -> {pool_path}")
    for source, n in report["rows_by_source"].items():
        print(f"  {source}: {n:,} pairs, "
              f"{report['distinct_candidates_by_source'][source]:,} candidates")
    cost = report["teacher_cost"]
    print(f"  teacher calls <= {cost['calls_upper_bound']:,} ({cost['calls_by_pass']})")
    print(f"funnel  -> {funnel_path}\nprompts -> {prompts_path}")


if __name__ == "__main__":
    main()
