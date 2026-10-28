#!/usr/bin/env python
"""Training rows for counterfactual decision distillation (docs/DECISION_DISTILL.md).

Every attributed row of grpo-v3-all's training split (`grpo-v3-all-free/grpo_train.parquet`,
the same 9,140 rows GRPO-improve trained on) gets its pair's attribute-free prompt, read from
the round-3 pools' `prompts.parquet` (kind `attr_free`, keyed by `pair_id`), as `free_prompt`.
Each pair then adds one row whose prompt is that attribute-free prompt itself (`is_free`), the
anchor that holds SFT-v3's attribute-free decision. No text is generated; the candidates are
grpo-v3-all's, none of them in the benchmark (checked again here against holdout_ids.json).

    python scripts/build_decision_distill_data.py \
        --rows artifacts/semisynthetic-v3/grpo-v3-all-free/grpo_train.parquet \
        --pool artifacts/semisynthetic-v3/onpolicy-new \
        --pool artifacts/semisynthetic-v3/onpolicy-new2 \
        --out artifacts/semisynthetic-v3/distill-v3-all
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KEEP = ["prompt", "free_prompt", "is_free", "pair_id", "candidate_id", "job_id",
        "reference_decision", "protected_group", "protected_attr", "condition", "tipped",
        "free_hire_rate"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def free_prompts(pools: list[Path]) -> pd.Series:
    parts = []
    for pool in pools:
        p = pd.read_parquet(pool / "prompts.parquet")
        parts.append(p[p["kind"] == "attr_free"].set_index("key")["prompt"])
    out = pd.concat(parts)
    if out.index.duplicated().any():
        raise ValueError("a pair has an attribute-free prompt in more than one pool")
    return out


def build(rows: pd.DataFrame, free: pd.Series) -> pd.DataFrame:
    attributed = rows.assign(free_prompt=rows["pair_id"].map(free), is_free=False)
    if attributed["free_prompt"].isna().any():
        raise ValueError(f"{int(attributed['free_prompt'].isna().sum())} rows lack an "
                         "attribute-free prompt")
    if (attributed["free_prompt"] == attributed["prompt"]).any():
        raise ValueError("an attributed prompt equals its attribute-free prompt")
    anchors = (attributed.drop_duplicates("pair_id")
               .assign(prompt=lambda f: f["free_prompt"], is_free=True,
                       protected_group="none", protected_attr="none", condition="attr_free",
                       tipped=False))
    return pd.concat([attributed, anchors], ignore_index=True)[KEEP]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--pool", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = pd.read_parquet(args.rows)
    out = build(rows, free_prompts(args.pool))
    holdout = json.loads((REPO / "data/benchmark/holdout_ids.json").read_text())
    held = set(holdout["candidate_ids"])
    if out["candidate_id"].isin(held).any() or out["job_id"].isin(set(holdout["job_ids"])).any():
        raise ValueError("a benchmark candidate or job is in the distillation rows")
    args.out.mkdir(parents=True, exist_ok=False)
    dest = args.out / "distill_train.parquet"
    out.to_parquet(dest, index=False)
    report = {"rows": len(out), "attributed": int((~out["is_free"]).sum()),
              "attribute_free": int(out["is_free"].sum()),
              "pairs": int(out["pair_id"].nunique()),
              "candidates": int(out["candidate_id"].nunique()),
              "holdout_checked": {"candidates": len(held), "jobs": len(holdout["job_ids"])},
              "source_rows": str(args.rows), "source_rows_sha256": sha(args.rows),
              "pools": [str(p) for p in args.pool], "sha256": sha(dest)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
