"""Pick the poster's dataset example from SFT-v3's audit and write ``poster/example.json``.

The rule, written after looking at SFT-v3's six unstable military-status sets (implicit
wording) to pick a short, clear one: military status, implicit wording; GPT-4o's
reference and SFT-v3's attribute-free decision both say hire; every military variant is
rejected and the civilian one hired; of those sets, the shortest CV. Everything shown is quoted
as written. Also counts the benchmark's attribute values per group.

Needs pandas (the shared venv), and the raw audit under ``$HBM_OUTPUT_ROOT/outputs/raw``:

    ../venv/bin/python poster/pick_example.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = "Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional"
GROUP, CONDITION, KEEP = "military_status", "implicit", "Civilian"
RULE = ("military status, implicit wording; GPT-4o's reference and SFT-v3's attribute-free "
        "decision both say hire; every military variant is rejected and the civilian one hired; "
        "of those sets, the shortest CV")


def main() -> None:
    root = Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
    df = pd.read_parquet(root / "outputs/raw" / f"{RUN}.parquet")

    free = df[df.condition == "attr_free"].set_index("pair_id").decision
    cands = []
    sets = df[(df.protected_group == GROUP) & (df.condition == CONDITION)]
    for pid, s in sets.groupby("pair_id"):
        others = s[s.protected_attr != KEEP]
        if (s.iloc[0].reference_decision == "hire" and free.get(pid) == "hire"
                and (s[s.protected_attr == KEEP].decision == "hire").all()
                and (others.decision == "reject").all()):
            cands.append((len(s.iloc[0].cv), pid))
    if not cands:
        raise SystemExit("no set matches the rule")
    pid = min(cands)[1]
    s = df[(df.pair_id == pid) & (df.protected_group == GROUP) & (df.condition == CONDITION)]
    first = s.iloc[0]

    req = [ln.strip("- ").strip() for ln in first.job_description.splitlines()
           if "must" in ln.lower()]
    cv = [ln.strip("- ").strip() for ln in first.cv.splitlines() if ln.strip()]
    variants = [{"value": r.protected_attr, "sentence": r.implicit_injection,
                 "decision": r.decision, "feedback": r.feedback}
                for r in s.itertuples()]
    groups = (df[df.condition != "attr_free"].groupby("protected_group").protected_attr
              .nunique().to_dict())

    out = {
        "source": f"outputs/raw/{RUN}.parquet",
        "rule": RULE,
        "matching_sets": len(cands),
        "pair_id": pid,
        "job_position": first.job_position,
        "job_musts": req,
        "cv_role": cv[1] if len(cv) > 1 else cv[0],
        "cv_lines": cv[4:6],
        "reference_decision": first.reference_decision,
        "attr_free_decision": free[pid],
        "variants": variants,
        "values_per_group": {k: int(v) for k, v in groups.items()},
    }
    path = HERE / "example.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path.name}: {first.job_position}, {len(cands)} matching sets")


if __name__ == "__main__":
    main()
