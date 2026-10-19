"""What is in a DPO preference set: rows by protected attribute, condition, reference decision
and source, and how long the two rationales are. CPU only, seconds.

The DPO sets start from v1's English invariant pool (`en_invariant.parquet`, the teacher's
attribute-free rationales). `--v1-invariant` rebuilds that pool with v1's own filter, exactly
as the on-policy sampler does (generation/onpolicy.pool_rows), and describes it. `--pairs`
describes one or more finished preference sets, each as its train and validation parquet.

    python scripts/describe_dpo_data.py \\
        --v1-invariant artifacts/semisynthetic-v1/raw/en_invariant.parquet \\
        --pairs r1=$R1/dpo_v2_train.parquet,$R1/dpo_v2_validation.parquet \\
        --pairs r2em=$R2EM/dpo_round2_train.parquet,$R2EM/dpo_round2_validation.parquet \\
        --out docs/dpo_data

with R1=artifacts/semisynthetic-v2/onpolicy-hire32/balanced and
R2EM=artifacts/semisynthetic-v2/round2-error-mass.

Writes, per described set and split: `<name>_attributes.csv` and `<name>_groups.csv` (rows
and reference decisions per attribute value and per group), `<name>_conditions.csv` (rows
per group and injection condition) and `<name>_summary.json` (sizes, decision shares,
sources, rationale lengths, columns). A row of
the pool is one CV and job with one attribute in one condition; a row of a preference set is
one (prompt, chosen, rejected) pair.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.data.protected_groups import GROUPS_IN_SCOPE, load_group  # noqa: E402
from hiring_bias_mitigation.eval.parsing import parse_output  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

DECISIONS = ("hire", "reject")


def _attribute_order(lang: str = "en") -> list[tuple[str, str]]:
    """(group, attribute) in the order of the attribute lists, so tables read like the audit's."""
    return [(group, attr) for group in GROUPS_IN_SCOPE
            for attr in load_group(group, lang).attributes]


def attribute_table(frame: pd.DataFrame, lang: str = "en") -> pd.DataFrame:
    """Rows per attribute value, split by the attribute-free reference decision."""
    counts = (frame.groupby(["protected_group", "protected_attr", "reference_decision"])
              .size().unstack(fill_value=0))
    for decision in DECISIONS:
        if decision not in counts.columns:
            counts[decision] = 0
    order = [key for key in _attribute_order(lang) if key in counts.index]
    order += [key for key in counts.index if key not in order]
    table = counts.loc[order, list(DECISIONS)].reset_index()
    table.insert(2, "rows", table["hire"] + table["reject"])
    return table.rename(columns={"protected_group": "group", "protected_attr": "attribute"})


def group_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows per protected group, split by the attribute-free reference decision."""
    table = attribute_table(frame).groupby("group", sort=False)[["rows", "hire", "reject"]].sum()
    return table.reset_index()


def condition_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows per group and injection condition (explicit field or first-person sentence)."""
    table = pd.crosstab(frame["protected_group"], frame["condition"])
    order = [g for g in GROUPS_IN_SCOPE if g in table.index]
    return table.loc[order + [g for g in table.index if g not in order]].reset_index().rename(
        columns={"protected_group": "group"})


def _quantiles(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {}
    q = values.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    return {"mean": round(float(values.mean()), 2), "min": float(q[0.0]), "p25": float(q[0.25]),
            "median": float(q[0.5]), "p75": float(q[0.75]), "max": float(q[1.0])}


def _feedback_words(texts: pd.Series, lang: str) -> pd.Series:
    return pd.Series([len(parse_output(t, lang).feedback.split()) for t in texts],
                     index=texts.index)


def pair_summary(frame: pd.DataFrame, lang: str = "en") -> dict:
    """Sizes, decision shares, sources and rationale lengths of one split of a preference set."""
    out: dict = {
        "preference_pairs": len(frame),
        "cv_job_pairs": int(frame["pair_id"].nunique()),
        "candidates": int(frame["candidate_id"].nunique()),
        "jobs": int(frame["job_id"].nunique()),
        "reference_decision": frame["reference_decision"].value_counts().to_dict(),
        "chosen_hire_share": round(float((frame["reference_decision"] == "hire").mean()), 4),
        "by_group": frame["protected_group"].value_counts().to_dict(),
        "by_condition": frame["condition"].value_counts().to_dict(),
    }
    if "source" in frame.columns:
        out["by_source"] = {
            str(s): part["reference_decision"].value_counts().to_dict()
            for s, part in frame.groupby("source")
        }
    if "rejected_strategy" in frame.columns:
        out["by_rejected_strategy"] = frame["rejected_strategy"].value_counts(dropna=False) \
            .rename(index=str).to_dict()
    chosen = _feedback_words(frame["chosen"], lang)
    rejected = _feedback_words(frame["rejected"], lang)
    out["chosen_words"] = _quantiles(chosen)
    out["rejected_words"] = _quantiles(rejected)
    out["rejected_minus_chosen_words"] = _quantiles(rejected - chosen)
    for column in ("judge_plausibility", "chosen_judge_plausibility", "token_overlap",
                   "free_right_rate", "n_wrong_samples", "student_margin", "p_wrong"):
        if column in frame.columns:
            out[column] = _quantiles(frame[column])
    out["columns"] = list(frame.columns)
    return out


def pool_summary(rows: pd.DataFrame, filter_report: dict) -> dict:
    """The v1 pool after v1's invariant filter: sizes, decisions, and what a pair looks like."""
    pairs = rows.drop_duplicates("pair_id")
    return {
        "invariant_filter": filter_report,
        "rows": len(rows),
        "cv_job_pairs": int(pairs["pair_id"].nunique()),
        "candidates": int(rows["candidate_id"].nunique()),
        "jobs": int(rows["job_id"].nunique()),
        "rows_per_pair": _quantiles(rows.groupby("pair_id").size()),
        "jobs_per_candidate": pairs.groupby("candidate_id").size().value_counts()
        .sort_index().rename(index=str).to_dict(),
        "reference_decision_rows": rows["reference_decision"].value_counts().to_dict(),
        "reference_decision_pairs": pairs["reference_decision"].value_counts().to_dict(),
        "primary_keywords": int(pairs["primary_keyword"].nunique()),
        "top_primary_keywords": pairs["primary_keyword"].value_counts().head(10).to_dict(),
        "candidate_years": _quantiles(pairs["candidate_years"]),
        "job_required_years": _quantiles(pairs["job_required_years"]),
        "cv_chars": _quantiles(pairs["cv"].str.len()),
        "job_description_chars": _quantiles(pairs["job_description"].str.len()),
        "chosen_feedback_words": _quantiles(rows["chosen_feedback"].str.split().str.len()),
    }


def _resolve(path: str) -> Path:
    local = Path(path)
    return local if local.exists() else Path(resolve_output_path(path))


def _provenance(path: str) -> dict:
    """The file's name and sha256: where it sat on this machine says nothing to a reader."""
    import hashlib

    resolved = _resolve(path)
    return {"file": resolved.name, "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()}


def _per_split(table, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([table(frame).assign(split=split) for split, frame in frames.items()],
                     ignore_index=True)


def _write(out: Path, name: str, frames: dict[str, pd.DataFrame], summary: dict) -> None:
    _per_split(attribute_table, frames).to_csv(out / f"{name}_attributes.csv", index=False)
    _per_split(group_table, frames).to_csv(out / f"{name}_groups.csv", index=False)
    _per_split(condition_table, frames).to_csv(out / f"{name}_conditions.csv", index=False)
    (out / f"{name}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(f"wrote {out}/{name}_{{attributes,groups,conditions}}.csv and {name}_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v1-invariant", help="v1's raw en_invariant.parquet")
    parser.add_argument("--pairs", action="append", default=[],
                        help="NAME=TRAIN.parquet[,VALIDATION.parquet]; repeatable")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.v1_invariant:
        from hiring_bias_mitigation.generation.onpolicy import pool_rows

        rows, report = pool_rows(pd.read_parquet(_resolve(args.v1_invariant)), args.lang)
        summary = pool_summary(rows, report)
        summary["source_file"] = _provenance(args.v1_invariant)
        _write(out, "v1_pool", {"pool": rows}, summary)

    for spec in args.pairs:
        name, _, paths = spec.partition("=")
        splits = dict(zip(("train", "validation"), paths.split(",")))
        frames = {split: pd.read_parquet(_resolve(path)) for split, path in splits.items()}
        summary = {split: pair_summary(frame, args.lang) for split, frame in frames.items()}
        summary["source_files"] = {split: _provenance(path) for split, path in splits.items()}
        _write(out, name, frames, summary)


if __name__ == "__main__":
    main()
