"""Round-2 DPO data: v2 pairs balanced by the SFT model's log-prob margin, not by count.

    python scripts/margin_balance.py prepare
    python scripts/margin_balance.py score --shard 0 --shards 4     # one per GPU
    python scripts/margin_balance.py select

  prepare  the train pairs available before count balancing: the full-pool run's split plus
           the hire top-up's new rows (onpolicy.merge_topup), -> OUT/pairs.parquet
  score    the SFT student's log-prob of chosen and rejected for this shard's pairs (HF + PEFT,
           the D4 measurement), -> OUT/scores/shard-N.parquet
  select   onpolicy.balance_by_margin on student_margin_sum; writes dpo_v2_train.parquet, the
           round-1 validation split unchanged (so both rounds evaluate on the same pairs),
           dpo_v2_candidates.parquet and balance.json under OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.generation import direction_check as DC  # noqa: E402
from hiring_bias_mitigation.generation import onpolicy as OP  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402
from hiring_bias_mitigation.utils.logging import get_logger  # noqa: E402

log = get_logger("margin_balance")

V2 = "artifacts/semisynthetic-v2"
OUT = f"{V2}/onpolicy-hire32/margin_balanced"
BASE_MODEL = "Qwen/Qwen3.5-9B"
SFT = "Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en"
SCORES = ["student_logp_chosen", "student_logp_rejected", "student_margin_sum",
          "student_margin_per_token", "student_decision_margin"]


def _p(rel: str) -> Path:
    return Path(resolve_output_path(rel))


def prepare(out: Path) -> None:
    base = pd.read_parquet(_p(f"{V2}/onpolicy-full/dpo_v2_train.parquet"))
    topup = pd.read_parquet(_p(f"{V2}/onpolicy-hire32/dpo_v2_train.parquet"))
    pairs, info = OP.merge_topup(base, topup)
    pairs.to_parquet(out / "pairs.parquet", index=False)
    log.info("%d train pairs to score (%s)", len(pairs), info)


def score(out: Path, shard: int, shards: int, batch_size: int) -> None:
    path = out / "scores" / f"shard-{shard}.parquet"
    if path.exists():
        log.info("%s exists; skipping", path)
        return
    pairs = pd.read_parquet(out / "pairs.parquet")
    part = pairs.iloc[shard::shards]
    scored, _ = DC.student_report(part, BASE_MODEL, SFT, batch_size=batch_size)
    path.parent.mkdir(exist_ok=True)
    scored[[*OP.ROW_KEY, *SCORES]].to_parquet(path, index=False)
    log.info("shard %d: %d pairs scored", shard, len(scored))


def select(out: Path, shards: int, bins: int, seed: int) -> None:
    pairs = pd.read_parquet(out / "pairs.parquet")
    files = [out / "scores" / f"shard-{s}.parquet" for s in range(shards)]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"missing score shards: {missing}")
    scores = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    scored = pairs.merge(scores, on=list(OP.ROW_KEY), how="inner", validate="one_to_one")
    assert len(scored) == len(pairs), (len(scored), len(pairs))
    train, info = OP.balance_by_margin(scored, "student_margin_sum", bins=bins, seed=seed)
    validation = pd.read_parquet(_p(f"{V2}/onpolicy-hire32/balanced/dpo_v2_validation.parquet"))
    train.to_parquet(out / "dpo_v2_train.parquet", index=False)
    validation.to_parquet(out / "dpo_v2_validation.parquet", index=False)
    pd.concat([train, validation], ignore_index=True).to_parquet(
        out / "dpo_v2_candidates.parquet", index=False)
    prefers = scored.assign(prefers=scored["student_margin_sum"] > 0).groupby(
        "reference_decision")["prefers"].mean()
    info["sft_prefers_chosen_before"] = {k: float(v) for k, v in prefers.items()}
    info["sft_prefers_chosen_after"] = {
        k: float(v) for k, v in (train["student_margin_sum"] > 0).groupby(
            train["reference_decision"]).mean().items()}
    info["validation"] = {"n": len(validation), "source": "onpolicy-hire32/balanced"}
    (out / "balance.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("prepare", "score", "select"))
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = _p(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        prepare(out)
    elif args.stage == "score":
        score(out, args.shard, args.shards, args.batch_size)
    else:
        select(out, args.shards, args.bins, args.seed)


if __name__ == "__main__":
    main()
