"""Claude's verdict on the anchor-reject / student-hire pairs that each training set actually used."""
import json, sys
from pathlib import Path
import pandas as pd
w = Path(sys.argv[1])
g = pd.read_csv(w / "graded_cases.csv")
p = pd.read_csv(w / "data_from_daint" / "pairs.csv")
cols = ["rows_in_dpo_r1", "rows_in_dpo_r2", "rows_in_dpo_r2em", "rows_in_grpo_02", "rows_in_rft_01"]
t = g[g.set == "target"].merge(p[["pair_id"] + cols], on="pair_id", how="left")
out = {}
for c in cols:
    u = t[t[c] > 0]
    out[c.replace("rows_in_", "")] = {
        "target_pairs_used": int(len(u)), "rows": int(u[c].sum()),
        "rows_claude_hire": int(u.loc[u.decision == "hire", c].sum()),
        "rows_claude_clear_hire": int(u.loc[u.fit >= 4, c].sum()),
        "rows_claude_borderline": int(u.loc[u.fit == 3, c].sum()),
        "rows_claude_clear_reject": int(u.loc[u.fit <= 2, c].sum()),
    }
(w / "by_training_set.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
