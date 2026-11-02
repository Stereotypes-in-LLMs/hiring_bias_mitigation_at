"""Before-and-after cases for the poster, from the raw audits, written to ``poster/cases.json``.

For each protected group, under implicit wording: the counterfactual sets the base model
decides inconsistently (every variant decided, not all the same) and SFT-v3 decides the same
way on every variant, agreeing with GPT-4o's reference and with its own attribute-free
decision. Of those, the set with the shortest CV is the case shown; the base model's two
decisions are quoted as written, with one feedback each, and SFT-v3's one decision with its
feedback. Beside the cases, the counts: how many of the base model's unstable implicit sets
SFT-v3 and self-distillation (seed 43) leave stable, per group.

Needs pandas (the shared venv), and the raw audits under ``$HBM_OUTPUT_ROOT/outputs/raw``:

    HBM_OUTPUT_ROOT=$XW ../venv/bin/python poster/pick_cases.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = {
    "Base": "Qwen3.5-9B--en--baseline--cscs",
    "SFT-v3": "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional",
    "Self-distillation s43": "Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed43",
}
GROUPS = ("military_status", "religion", "gender")
CONDITION = "implicit"
RULE = ("implicit wording; the base model decides the set's variants differently (all decided); "
        "SFT-v3 decides every variant the same way, as GPT-4o's reference and as its own "
        "attribute-free decision; of those sets, the shortest CV")
# The poster shows the gender case: its injected sentence states pronouns and nothing else,
# whereas the military ("participating in combat actions") and religion ("would need days off")
# sentences also carry availability, which a reader could take as a legitimate reason.
POSTER_CASE = "gender"
DATES = re.compile(r"^\d{2}\.\d{2}\.\d{4}\s*[-–]\s*\d{2}\.\d{2}\.\d{4}")


def cv_evidence(cv: str) -> dict | None:
    """The CV's first dated position: dates, company, position, what it was."""
    lines = [ln.strip() for ln in cv.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if DATES.match(ln) and i + 3 < len(lines) and lines[i + 2].startswith("Position:"):
            return {"dates": ln, "company": lines[i + 1],
                    "position": lines[i + 2].removeprefix("Position:").strip(),
                    "detail": lines[i + 3]}
    return None


def requirement(job_description: str) -> str | None:
    """The job's first experience requirement line, bullet stripped."""
    for ln in job_description.splitlines():
        if re.search(r"\d\+?\s*years?", ln, re.I):
            return ln.strip().lstrip("•-* \t")
    return None


def sets_of(df: pd.DataFrame, group: str) -> pd.core.groupby.DataFrameGroupBy:
    return df[(df.protected_group == group) & (df.condition == CONDITION)].groupby("pair_id")


def unstable(s: pd.DataFrame) -> bool:
    return (s.outcome == "decided").all() and s.decision.nunique() > 1


def stable_as(s: pd.DataFrame, decision: str) -> bool:
    return (s.outcome == "decided").all() and (s.decision == decision).all()


def main() -> None:
    root = Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
    raw = {k: pd.read_parquet(root / "outputs/raw" / f"{v}.parquet") for k, v in RUNS.items()}
    base = raw["Base"]
    free = {k: d[d.condition == "attr_free"].set_index("pair_id").decision for k, d in raw.items()}
    cases, counts = [], {}
    for group in GROUPS:
        base_sets = {pid: s for pid, s in sets_of(base, group) if unstable(s)}
        after = {k: {pid: s for pid, s in sets_of(d, group)} for k, d in raw.items() if k != "Base"}
        counts[group] = {
            "base_unstable_sets": len(base_sets),
            **{f"stable_in_{k}": sum(
                (pid in sets and (sets[pid].outcome == "decided").all()
                 and sets[pid].decision.nunique() == 1) for pid in base_sets)
               for k, sets in after.items()},
        }
        cands = []
        for pid, s in base_sets.items():
            t = after["SFT-v3"].get(pid)
            if t is None:
                continue
            ref = s.iloc[0].reference_decision
            if free["SFT-v3"].get(pid) == ref and stable_as(t, ref):
                cands.append((len(s.iloc[0].cv), pid))
        if not cands:
            continue
        pid = min(cands)[1]
        s, t = base_sets[pid], after["SFT-v3"][pid]
        first = s.iloc[0]
        cv = [ln.strip("- ").strip() for ln in first.cv.splitlines() if ln.strip()]

        def quote(rows: pd.DataFrame, decision: str) -> dict:
            r = rows[rows.decision == decision].iloc[0]
            return {"value": r.protected_attr, "sentence": r.implicit_injection,
                    "decision": r.decision, "feedback": r.feedback}

        cases.append({
            "group": group, "pair_id": pid, "matching_sets": len(cands),
            "job_position": first.job_position, "cv_role": cv[1] if len(cv) > 1 else cv[0],
            "job_requirement": requirement(first.job_description),
            "cv_evidence": cv_evidence(first.cv),
            "reference_decision": first.reference_decision,
            "base": {"decisions": s.set_index("protected_attr").decision.to_dict(),
                     "hire": quote(s, "hire"), "reject": quote(s, "reject"),
                     "attr_free": free["Base"].get(pid)},
            "sft_v3": {"decision": t.iloc[0].decision, "feedback": t.iloc[0].feedback,
                       "attr_free": free["SFT-v3"].get(pid)},
            "self_distillation_s43": {
                "decisions": after["Self-distillation s43"][pid].set_index("protected_attr")
                .decision.to_dict()} if pid in after["Self-distillation s43"] else None,
        })
    out = {"sources": {k: f"outputs/raw/{v}.parquet" for k, v in RUNS.items()},
           "rule": RULE, "condition": CONDITION, "poster_case": POSTER_CASE,
           "counts": counts, "cases": cases}
    path = HERE / "cases.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(counts, indent=1))
    for c in cases:
        print(f"{c['group']}: {c['job_position']} | base {c['base']['decisions']} | "
              f"SFT-v3 {c['sft_v3']['decision']} (ref {c['reference_decision']}); "
              f"{c['matching_sets']} matching sets")


if __name__ == "__main__":
    main()
