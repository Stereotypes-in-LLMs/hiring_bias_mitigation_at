"""Builds the GRPO pilot's prompt set (CPU, seconds). What it keeps and why:
`src/hiring_bias_mitigation/generation/grpo_prompts.py`.

    python scripts/build_grpo_prompts.py                     # grpo-02: both sides the same pull
    python scripts/build_grpo_prompts.py --balance tipped --tipped-share 0.5 \
        --out artifacts/semisynthetic-v2/grpo-01            # grpo-01: half of each side tipped

Inputs are the v1 English pool as the SFT student was sampled on it (`onpolicy-full`: rows,
their attributed prompts, the k=8 selection and samples) and the k=32 hire top-up's selection
and samples (`onpolicy-hire32`). Every candidate in a `*validation.parquet` under
semisynthetic-v2 is held out of training. Output, read by a training config with `data_kind: grpo`:

    grpo_train.parquet, grpo_validation.parquet, report.json, samples.md
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
from hiring_bias_mitigation.utils.config import (  # noqa: E402
    require_output_root,
    resolve_output_path,
)
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("build_grpo_prompts")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="artifacts/semisynthetic-v2/onpolicy-full")
    parser.add_argument("--hire-topup", default="artifacts/semisynthetic-v2/onpolicy-hire32")
    parser.add_argument("--validation-root", default="artifacts/semisynthetic-v2",
                        help="every *validation.parquet below it names held-out candidates")
    parser.add_argument("--out", default="artifacts/semisynthetic-v2/grpo-02")
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--balance", choices=("signal", "tipped"), default="signal",
                        help="signal: both halves the same summed pull (grpo-02); tipped: "
                             "--tipped-share of each half from tipped rows (grpo-01)")
    parser.add_argument("--tipped-share", type=float, default=0.5)
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
    rows = rows[rows["lang"] == "en"]
    keys = G.tipped_keys(G.read(pool / "selected.parquet"), G.read(topup / "selected.parquet"))
    rows = G.mark_tipped(rows, keys)

    def counts(root: Path) -> dict:
        chunks = sorted((root / "raw" / "samples").glob("chunk-*.parquet"))
        frame = pd.concat([G.read(c, ["prompt", "raw_output"]) for c in chunks])
        return G.sample_hire_counts(frame[frame["prompt"].isin(set(rows["prompt"]))])

    # The k=32 top-up first: its hire rows' rates rest on four times the samples.
    rows = G.attach_sft_rates(rows, counts(topup), counts(pool))

    root = Path(resolve_output_path(args.validation_root))
    split_files = sorted(p for p in root.rglob("*validation.parquet") if out not in p.parents)
    heldout = G.heldout_candidates([G.read(p, ["candidate_id"]) for p in split_files])
    train_pool, held_pool = G.split_by_candidate(rows, heldout)
    log.info("%d held-out candidates from %d validation splits: %d rows held out, %d left",
             len(heldout), len(split_files), len(held_pool), len(train_pool))

    if args.balance == "signal":
        train, plan = G.signal_balanced_train(train_pool, args.n_train, args.seed)
    else:
        train, plan = G.balanced_train(train_pool, args.n_train, args.tipped_share, args.seed)
    validation = G.balanced_validation(held_pool, args.n_validation, args.seed)
    assert_no_leakage(train, "grpo_train")
    assert_no_leakage(validation, "grpo_validation")
    overlap = set(train["candidate_id"].astype(str)) & set(validation["candidate_id"].astype(str))
    if overlap:
        raise AssertionError(f"{len(overlap)} candidates in both GRPO splits")

    out.mkdir(parents=True, exist_ok=True)
    train[G.COLUMNS].to_parquet(out / "grpo_train.parquet", index=False)
    validation[G.COLUMNS].to_parquet(out / "grpo_validation.parquet", index=False)
    report = {
        "inputs": {"pool": args.pool, "hire_topup": args.hire_topup,
                   "validation_splits": [str(p.relative_to(root)) for p in split_files]},
        "settings": {"n_train": args.n_train, "balance": args.balance,
                     "tipped_share": args.tipped_share if args.balance == "tipped" else None,
                     "n_validation": args.n_validation, "seed": args.seed},
        "pool": G.describe(rows),
        "heldout": {"candidates": len(heldout), "rows": len(held_pool)},
        "draw": plan,
        "train": G.describe(train),
        "train_pull": G.side_pull(train),
        "validation": G.describe(validation),
        "leakage": "assert_no_leakage passed on both splits; no candidate in both",
        "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "samples.md").write_text(G.samples_markdown(train, seed=args.seed))
    log.info("wrote %s: %d train, %d validation prompts", out, len(train), len(validation))
    print(json.dumps({k: report[k] for k in ("draw", "train_pull", "train")}, indent=2))


if __name__ == "__main__":
    main()
