"""Builds the rejection-sampling SFT control's data (CPU, a few minutes). What it keeps and why:
`src/hiring_bias_mitigation/generation/rft_data.py`.

    python scripts/build_rft_data.py
    python scripts/build_rft_data.py --n-train 2400 --out artifacts/semisynthetic-v2/rft-01

Inputs are the v1 English pool as the SFT student was sampled on it (`onpolicy-full`: rows,
their attributed prompts, the k=8 selection and samples) and the k=32 hire top-up's selection
and samples (`onpolicy-hire32`). Every candidate in a `*validation.parquet` under
semisynthetic-v2 is held out of training. Output, read by a training config with
`data_kind: rft`:

    rft_train.parquet, rft_validation.parquet, report.json, samples.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402

from hiring_bias_mitigation.data.benchmark import assert_no_leakage  # noqa: E402
from hiring_bias_mitigation.generation import grpo_prompts as G  # noqa: E402
from hiring_bias_mitigation.generation import rft_data as R  # noqa: E402
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("build_rft_data")


def read_samples(root: Path, prompts: set[str]) -> pd.DataFrame:
    chunks = sorted((root / "raw" / "samples").glob("chunk-*.parquet"))
    frame = pd.concat([G.read(c, ["prompt", "sample_index", "raw_output"]) for c in chunks])
    return frame[frame["prompt"].isin(prompts)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="artifacts/semisynthetic-v2/onpolicy-full")
    parser.add_argument("--hire-topup", default="artifacts/semisynthetic-v2/onpolicy-hire32")
    parser.add_argument("--validation-root", default="artifacts/semisynthetic-v2",
                        help="every *validation.parquet below it names held-out candidates")
    parser.add_argument("--out", default="artifacts/semisynthetic-v2/rft-01")
    parser.add_argument("--n-train", type=int, default=2400)
    parser.add_argument("--n-validation", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)

    require_output_root()
    pool = Path(resolve_output_path(args.pool))
    topup = Path(resolve_output_path(args.hire_topup))
    out = Path(resolve_output_path(args.out))

    rows = G.attach_prompts(G.read(pool / "rows.parquet"), G.read(pool / "prompts.parquet"))
    rows = rows[rows["lang"] == "en"].reset_index(drop=True)
    rows = G.mark_tipped(rows, G.tipped_keys(G.read(pool / "selected.parquet"),
                                             G.read(topup / "selected.parquet")))
    prompts = set(rows["prompt"])
    k32, k8 = read_samples(topup, prompts), read_samples(pool, prompts)
    # The k=32 top-up first, for the rates and for the target: four times the samples.
    rows = G.attach_sft_rates(rows, G.sample_hire_counts(k32), G.sample_hire_counts(k8))
    found32, report32 = R.first_right_samples(rows, k32, "onpolicy-hire32")
    found8, report8 = R.first_right_samples(rows, k8, "onpolicy-full")
    targets = R.attach_targets(rows, found32, found8)
    log.info("%d of %d rows have a right sample that passes the filters", len(targets), len(rows))

    root = Path(resolve_output_path(args.validation_root))
    split_files = sorted(p for p in root.rglob("*validation.parquet") if out not in p.parents)
    heldout = G.heldout_candidates([G.read(p, ["candidate_id"]) for p in split_files])
    train_pool, held_pool = G.split_by_candidate(targets, heldout)

    train, plan = G.signal_balanced_train(train_pool, args.n_train, args.seed, column="error")
    validation = G.balanced_validation(held_pool, args.n_validation, args.seed)
    assert_no_leakage(train, "rft_train")
    assert_no_leakage(validation, "rft_validation")
    overlap = set(train["candidate_id"].astype(str)) & set(validation["candidate_id"].astype(str))
    if overlap:
        raise AssertionError(f"{len(overlap)} candidates in both RFT splits")

    out.mkdir(parents=True, exist_ok=True)
    train[R.COLUMNS].to_parquet(out / "rft_train.parquet", index=False)
    validation[R.COLUMNS].to_parquet(out / "rft_validation.parquet", index=False)
    report = {
        "inputs": {"pool": args.pool, "hire_topup": args.hire_topup,
                   "validation_splits": [str(p.relative_to(root)) for p in split_files]},
        "settings": {"n_train": args.n_train, "n_validation": args.n_validation,
                     "seed": args.seed, "balance": "equal counts and equal error mass"},
        "filters": {"onpolicy-hire32": report32, "onpolicy-full": report8},
        "pool": {"rows": len(rows), "rows_with_a_target": len(targets),
                 "rows_without_by_reference": {
                     str(k): int(v) for k, v in
                     rows[~rows["prompt"].isin(set(targets["prompt"]))]
                     ["reference_decision"].value_counts().items()}},
        "heldout": {"candidates": len(heldout), "rows": len(held_pool)},
        "draw": plan,
        "train": G.describe(train),
        "train_error_mass": G.side_pull(train),
        "train_target_source": {str(k): int(v) for k, v in
                                train["sample_source"].value_counts().items()},
        "validation": G.describe(validation),
        "leakage": "assert_no_leakage passed on both splits; no candidate in both",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "samples.md").write_text(R.samples_markdown(train, seed=args.seed))
    log.info("wrote %s: %d train, %d validation rows", out, len(train), len(validation))
    print(json.dumps({k: report[k] for k in ("pool", "draw", "train_error_mass",
                                             "train_target_source")}, indent=2))


if __name__ == "__main__":
    main()
