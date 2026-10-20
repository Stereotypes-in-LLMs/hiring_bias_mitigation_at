"""Builds shuffled blind grading batches and a sealed key for the anchor-vs-student review."""
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd

work = Path(sys.argv[1]); data = work / "data_from_daint"
rng = np.random.default_rng(42)
pairs = pd.read_csv(data / "pairs.csv")
pr = pd.read_parquet(data / "prompts.parquet")
af = pr[pr["kind"] == "attr_free"].set_index("key")["prompt"]

target = pairs[(pairs.anchor == "reject") & (pairs.student_majority == "hire")].assign(set="target")
reverse = pairs[(pairs.anchor == "hire") & (pairs.student_majority == "reject")].assign(set="reverse")
ah = pairs[(pairs.anchor == "hire") & (pairs.student_majority == "hire")]
ar = pairs[(pairs.anchor == "reject") & (pairs.student_majority == "reject")]
ctrl_h = ah.iloc[rng.choice(len(ah), 20, replace=False)].assign(set="control_hire")
ctrl_r = ar.iloc[rng.choice(len(ar), 20, replace=False)].assign(set="control_reject")
sel = pd.concat([target, reverse, ctrl_h, ctrl_r], ignore_index=True)
assert len(target) == 337 and len(reverse) == 14, (len(target), len(reverse))

pat = re.compile(r"Job description: ```\n(.*?)```\s*Candidate profile: ```\n(.*?)```", re.S)
jd, cv = [], []
for pid in sel["pair_id"]:
    m = pat.search(af[pid])
    assert m, pid
    jd.append(m.group(1).strip()); cv.append(m.group(2).strip())
sel["job_description"], sel["cv"] = jd, cv

sel = sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)
sel["item"] = [f"C{i+1:03d}" for i in range(len(sel))]
key_cols = ["item", "pair_id", "set", "anchor", "student_majority", "student_hire_share",
            "n_samples", "n_hire", "sft_heldout", "k32_hire_share"]
(work / "sealed").mkdir(exist_ok=True)
sel[key_cols].to_csv(work / "sealed" / "key.csv", index=False)

n_batches = 8
for b, idx in enumerate(np.array_split(np.arange(len(sel)), n_batches), start=1):
    chunk = sel.iloc[idx]
    lines = [f"# Batch {b:02d}: {len(chunk)} cases\n"]
    for _, r in chunk.iterrows():
        lines.append(f"## {r['item']}\n\nJob description:\n```\n{r['job_description']}\n```\n\n"
                     f"Candidate profile:\n```\n{r['cv']}\n```\n")
    (work / "batches" / f"batch_{b:02d}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"batch_{b:02d}: {len(chunk)} cases, {sum(len(x)+len(y) for x, y in zip(chunk.job_description, chunk.cv)):,} chars, items {chunk.item.iloc[0]}..{chunk.item.iloc[-1]}")
print("sets:", sel["set"].value_counts().to_dict())
