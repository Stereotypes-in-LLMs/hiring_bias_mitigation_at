"""Rejection-sampling SFT (ReST-style) targets: the student's own right answers, kept as data.

The control for the optimizer question. Three DPO rounds and a GRPO pilot continued from the
SFT model all landed near 1 pp at matched hire rate (docs/ITERATIVE_DPO.md,
docs/GRPO_PILOT.md). A second plain SFT stage on the student's own correct samples changes the
objective and keeps the anchors: if it lands at the same place, the ceiling is the data.

No generation. The SFT student already answered every v1 English pool row 8 times at
temperature 1 (`onpolicy-full/raw/samples`) and every hire-reference row 32 times
(`onpolicy-hire32/raw/samples`). A row's target is the first of its samples, in sample order,
that reaches the attribute-free reference and passes the hard filters the v1 targets passed
(`filters.filter_invariant`: parseable, right verdict, 4-45 words, right language, no attribute
named, no meta-commentary) plus the v2 group filter (`negatives_v2.mentions_group`). Sample
order, not a score, picks it, as for the on-policy negatives (generation/onpolicy.py).

Balance. Counts are equal per reference verdict, and so is error mass, as in r2em: a row on
which the student errs with probability p moves its decision by about p under the NLL of a
right answer, and attribute-tipped rows carry most of it. The weaker half takes every tipped
row it can, the stronger as many as match it (grpo_prompts.signal_balanced_train on `error`).
"""

from __future__ import annotations

import pandas as pd

from ..utils.logging import get_logger
from .dataset import _completion
from .filters import filter_invariant
from .negatives_v2 import mentions_group

log = get_logger(__name__)

#: Written next to `prompt` and `completion`; the training loader keeps what it needs.
COLUMNS = ["prompt", "completion", "decision", "lang", "protected_group", "protected_attr",
           "condition", "candidate_id", "job_id", "pair_id", "reference_decision", "tipped",
           "sft_hire_rate", "sft_samples", "error", "sample_index", "sample_source"]


def first_right_samples(rows: pd.DataFrame, samples: pd.DataFrame, source: str,
                        lang: str = "en") -> tuple[pd.DataFrame, dict]:
    """For each row with samples here, its first sample (by `sample_index`) that passes the
    filters, as `decision`/`feedback`/`sample_index`; rows with none are absent.

    `samples` holds the student's stored samples (`prompt`, `sample_index`, `raw_output`);
    a sample belongs to the row whose attributed prompt it answered.
    """
    index = pd.Series(range(len(rows)), index=rows["prompt"].to_numpy())
    if index.index.duplicated().any():
        raise ValueError("two pool rows share an attributed prompt; samples cannot be assigned")
    mine = samples[samples["prompt"].isin(index.index)]
    long = pd.DataFrame({
        "_row": index.loc[mine["prompt"]].to_numpy(),
        "sample_index": mine["sample_index"].astype(int).to_numpy(),
    })
    meta = rows[["reference_decision", "protected_attr", "protected_group"]].reset_index(drop=True)
    long = long.join(meta, on="_row")
    kept, report = filter_invariant(mine["raw_output"].tolist(), long, lang)
    if kept.empty:
        return kept, {**report, "group_leak": 0, "rows_with_a_target": 0}
    leak = [mentions_group(f, g) for f, g in zip(kept["chosen_feedback"], kept["protected_group"])]
    kept = kept[[not x for x in leak]]
    first = (kept.sort_values(["_row", "sample_index"]).drop_duplicates("_row", keep="first")
                 .rename(columns={"chosen_decision": "decision", "chosen_feedback": "feedback"}))
    report = {**report, "group_leak": int(sum(leak)), "rows_with_a_target": len(first),
              "rows_with_samples": int(long["_row"].nunique()), "source": source}
    return first[["_row", "sample_index", "decision", "feedback"]].assign(sample_source=source), \
        report


def attach_targets(rows: pd.DataFrame, *found: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Rows with the first target found, in the order the tables are given (pass the k=32
    top-up first), as an SFT `completion`; rows with no target are dropped."""
    taken = pd.concat(found).drop_duplicates("_row", keep="first").set_index("_row")
    out = rows.reset_index(drop=True).join(taken, how="inner")
    completion = [_completion(d, f, lang) for d, f in zip(out["decision"], out["feedback"])]
    error = (1 - out["sft_hire_rate"]).where(out["reference_decision"] == "hire",
                                             out["sft_hire_rate"])
    return out.assign(completion=completion, error=error).reset_index(drop=True)


def samples_markdown(frame: pd.DataFrame, n: int = 20, seed: int = 42,
                     head_chars: int = 300) -> str:
    """Twenty rows as the trainer sees them: the end of the prompt and the target."""
    lines = ["# RFT target samples", ""]
    for side in ("hire", "reject"):
        part = frame[frame["reference_decision"] == side]
        for row in part.sample(n=min(n // 2, len(part)), random_state=seed).to_dict("records"):
            lines += [f"## {side} | {row['protected_group']} | {row['protected_attr']} | "
                      f"{row['condition']} | tipped={row['tipped']} | student hires "
                      f"{row['sft_hire_rate']:.2f} of {row['sft_samples']} | sample "
                      f"{row['sample_source']}#{row['sample_index']}",
                      "", "```text", "..." + row["prompt"][-head_chars:], "```", "",
                      "```json", row["completion"], "```", ""]
    return "\n".join(lines)
