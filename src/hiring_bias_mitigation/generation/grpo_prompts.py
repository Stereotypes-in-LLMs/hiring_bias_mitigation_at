"""The GRPO pilot's prompt set: attributed audit prompts, balanced by reference verdict.

GRPO needs no responses, only prompts and what a right answer is. Each row here is one v1 pool
row (a pair's CV and job with one attribute in one condition) as the audit prompt the SFT
student was sampled on, with the pair's attribute-free reference verdict as the target. The
reward (mitigation/grpo.py) reads `reference_decision` next to each completion.

Three choices, each against a failure seen in the DPO rounds:

  Balance. The pool's reference is 86% reject. DPO on pool-shaped data taught "reject" and
  moved the hire rate from 28% to 19% (docs/ITERATIVE_DPO.md). Here both verdicts get exactly
  half the prompts, so a policy that drifts to one verdict loses reward on half of them.

  Tipped rows. A GRPO group whose completions all earn the same reward has zero advantage and
  teaches nothing. Rows where the student is right without the attribute and wrong with it in
  at least one sample (`tipped` in the on-policy selection) are where its samples disagree,
  so `tipped_share` of each side comes from them -- far above their natural rate. Hire rows
  read the k=32 hire top-up's selection, reject rows the k=8 pool's.

  Hold-out. Every candidate in an existing validation split stays out of training, so those
  splits stay comparable across arms, and the decision monitor reads its rows from them. Both
  frames pass `assert_no_leakage` against the benchmark, as every data builder here does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

#: One pool row: a pair's CV and job with one attribute in one condition.
ROW_KEY = ["pair_id", "protected_attr", "condition"]
#: Carried into the parquet next to `prompt`; the reward functions receive every one of them.
COLUMNS = ["prompt", "reference_decision", "pair_id", "candidate_id", "job_id", "lang",
           "protected_group", "protected_attr", "condition", "tipped"]
SIDES = ("hire", "reject")


def attach_prompts(rows: pd.DataFrame, prompts: pd.DataFrame) -> pd.DataFrame:
    """Each pool row with the attributed audit prompt it was sampled on.

    prompts.parquet keys an attributed prompt by its row's position in rows.parquet, so a row
    count that differs means the two files are not from the same run.
    """
    attributed = prompts[prompts["kind"] == "attributed"]
    if len(attributed) != len(rows):
        raise ValueError(f"{len(attributed)} attributed prompts for {len(rows)} rows: "
                         "rows.parquet and prompts.parquet are from different runs")
    text = dict(zip(attributed["key"].astype(int), attributed["prompt"]))
    return rows.reset_index(drop=True).assign(prompt=[text[i] for i in range(len(rows))])


def tipped_keys(*selections: pd.DataFrame) -> set[tuple]:
    """Row keys the on-policy selection marked attribute-tipped, over any number of runs."""
    keys: set[tuple] = set()
    for frame in selections:
        tipped = frame[frame["tipped"].astype(bool)]
        keys.update(tipped[ROW_KEY].itertuples(index=False, name=None))
    return keys


def mark_tipped(rows: pd.DataFrame, keys: set[tuple]) -> pd.DataFrame:
    flags = [k in keys for k in rows[ROW_KEY].itertuples(index=False, name=None)]
    return rows.assign(tipped=flags)


def heldout_candidates(frames: list[pd.DataFrame]) -> set[str]:
    return {str(c) for f in frames for c in f["candidate_id"]}


def split_by_candidate(rows: pd.DataFrame, heldout: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    held = rows["candidate_id"].astype(str).isin(heldout)
    return rows[~held].reset_index(drop=True), rows[held].reset_index(drop=True)


def _draw(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if n <= 0 or frame.empty:
        return frame.iloc[:0]
    index = rng.choice(len(frame), size=min(n, len(frame)), replace=False)
    return frame.iloc[np.sort(index)]


def balanced_train(rows: pd.DataFrame, n: int, tipped_share: float,
                   seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """`n` prompts, exactly half per reference verdict; within a side, `tipped_share` of the
    half from tipped rows and the rest from the others, each drawn without replacement. A side
    short of tipped rows fills up with untipped ones, so the halves stay exact."""
    if n % 2:
        raise ValueError(f"n={n} cannot split exactly in half")
    rng = np.random.default_rng(seed)
    half, parts, plan = n // 2, [], {}
    for side in SIDES:
        pool = rows[rows["reference_decision"] == side]
        tipped, rest = pool[pool["tipped"]], pool[~pool["tipped"]]
        want = int(round(tipped_share * half))
        a = _draw(tipped, want, rng)
        b = _draw(rest, half - len(a), rng)
        if len(a) + len(b) < half:
            raise ValueError(f"only {len(pool)} {side}-reference rows for a half of {half}")
        parts += [a, b]
        plan[side] = {"available": len(pool), "available_tipped": len(tipped),
                      "drawn_tipped": len(a), "drawn_untipped": len(b)}
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out, plan


def balanced_validation(rows: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """`n` held-out rows, half per reference verdict, one seeded draw."""
    rng = np.random.default_rng(seed)
    parts = [_draw(rows[rows["reference_decision"] == side], n // 2, rng) for side in SIDES]
    return pd.concat(parts).sort_index().reset_index(drop=True)


def describe(frame: pd.DataFrame) -> dict:
    """Counts a reader checks first: per verdict, tipped, group and condition."""
    def counts(column: str) -> dict:
        return {str(k): int(v) for k, v in frame[column].value_counts().sort_index().items()}

    by_side = frame.groupby("reference_decision")
    return {
        "rows": len(frame),
        "pairs": int(frame["pair_id"].nunique()),
        "candidates": int(frame["candidate_id"].nunique()),
        "reference_decision": counts("reference_decision"),
        "tipped_share_by_reference": {str(k): round(float(g["tipped"].mean()), 4)
                                      for k, g in by_side},
        "protected_group": counts("protected_group"),
        "condition": counts("condition"),
        "prompt_chars": {q: int(frame["prompt"].str.len().quantile(p))
                         for q, p in (("p50", 0.5), ("p99", 0.99), ("max", 1.0))},
    }


def samples_markdown(frame: pd.DataFrame, n: int = 12, seed: int = 42,
                     head_chars: int = 600) -> str:
    """A few rows per verdict as they reach the trainer, for reading before training."""
    lines = ["# GRPO prompt samples", ""]
    for side in SIDES:
        side_rows = frame[frame["reference_decision"] == side]
        pick = side_rows.sample(n=min(n // 2, len(side_rows)), random_state=seed)
        for row in pick.to_dict("records"):
            lines += [f"## {side} | {row['protected_group']} | {row['protected_attr']} | "
                      f"{row['condition']} | tipped={row['tipped']}",
                      f"pair `{row['pair_id']}`", "", "```text",
                      row["prompt"][:head_chars] + (" ..." if len(row["prompt"]) > head_chars
                                                     else ""),
                      "...", row["prompt"][-head_chars:], "```", ""]
    return "\n".join(lines)


def read(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Single-threaded parquet read: Daint login nodes refuse pyarrow's worker threads."""
    return pd.read_parquet(path, columns=columns, use_threads=False)
