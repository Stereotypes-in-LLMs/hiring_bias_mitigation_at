"""Validates the blind grades, joins them to the sealed key and scores the anchor review.

    python score.py <work_dir>

Reads grades/batch_*.csv and sealed/key.csv. Writes graded_cases.csv and results.json.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

work = Path(sys.argv[1])
key = pd.read_csv(work / "sealed" / "key.csv")

frames = []
problems = []
for path in sorted((work / "grades").glob("batch_*.csv")):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    df = pd.DataFrame(rows)
    df["batch"] = path.stem
    frames.append(df)
grades = pd.concat(frames, ignore_index=True)
grades.columns = [c.strip().lower() for c in grades.columns]
grades["item"] = grades["item"].str.strip()
grades["decision"] = grades["decision"].str.strip().str.lower()
grades["fit"] = pd.to_numeric(grades["fit"], errors="coerce")

dupes = grades[grades.duplicated("item", keep=False)]
if len(dupes):
    problems.append(f"duplicate items: {sorted(dupes.item.unique())}")
missing = sorted(set(key.item) - set(grades.item))
extra = sorted(set(grades.item) - set(key.item))
bad_decision = grades[~grades.decision.isin(["hire", "reject"])]
bad_fit = grades[~grades.fit.isin([1, 2, 3, 4, 5])]
inconsistent = grades[((grades.fit <= 2) & (grades.decision == "hire"))
                      | ((grades.fit >= 4) & (grades.decision == "reject"))]
for label, frame in (("bad decision", bad_decision), ("bad fit", bad_fit),
                     ("fit/decision inconsistent", inconsistent)):
    if len(frame):
        problems.append(f"{label}: {frame.item.tolist()}")

merged = key.merge(grades.drop_duplicates("item"), on="item", how="left")
merged["claude_hire"] = (merged.decision == "hire").astype(float).where(merged.decision.notna())
merged["fit_bucket"] = pd.cut(merged.fit, [0, 2, 3, 5], labels=["1-2", "3", "4-5"])
merged.to_csv(work / "graded_cases.csv", index=False)

rng = np.random.default_rng(42)


def share_ci(values, draws=2000):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"n": 0}
    boots = rng.choice(v, size=(draws, len(v)), replace=True).mean(axis=1)
    return {"n": int(len(v)), "share": float(v.mean()),
            "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975))}


out = {"validation": {"graded": int(merged.decision.notna().sum()), "expected": int(len(key)),
                      "missing": missing, "extra": extra, "problems": problems}}

by_set = {}
for name, g in merged.groupby("set"):
    shared = {"target": None, "reverse": None, "control_hire": "hire", "control_reject": "reject"}[name]
    entry = {"claude_hire": share_ci(g.claude_hire),
             "fit_counts": {str(k): int(v) for k, v in g.fit_bucket.value_counts().sort_index().items()},
             "decision_by_fit": {f"{b}/{d}": int(n) for (b, d), n in
                                 g.groupby(["fit_bucket", "decision"], observed=True).size().items()}}
    if shared:
        entry["agree_with_shared"] = share_ci((g.decision == shared).astype(float).where(g.decision.notna()))
    by_set[name] = entry
out["by_set"] = by_set

controls = merged[merged.set.str.startswith("control")]
shared = controls.set.map({"control_hire": "hire", "control_reject": "reject"})
out["controls_agreement"] = share_ci((controls.decision == shared).astype(float).where(controls.decision.notna()))

t = merged[merged.set == "target"].copy()
t["student_conf"] = pd.cut(t.student_hire_share, [0, 0.625, 0.875, 1.0],
                           labels=["5/8", "6-7/8", "8/8"], include_lowest=True)
out["target_by_student_confidence"] = {str(k): share_ci(g.claude_hire)
                                       for k, g in t.groupby("student_conf", observed=True)}
out["target_by_sft_heldout"] = {str(k): share_ci(g.claude_hire) for k, g in t.groupby("sft_heldout")}

(work / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
