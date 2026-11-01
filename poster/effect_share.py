"""The share of the base model's per-pair attribute effect that SFT-v3 removes, on all 450 pairs.

Marta Samoilenko's patching measured it on 64 pairs per contrast, picked for a large base effect
(docs/INTERP_SFT_V3.md: 61-77%). Here it is read on every aligned pair of the same three
contrasts, from the patching pipeline's manifest (scripts/interp/04_manifest.py): one minus
sum |dm_SFT-v3| over sum |dm_base|. Writes ``poster/effect_share.json``, which
``build_poster.py`` reads for DETAILS.md (the LaTeX poster quotes the patched-pairs share).

    HBM_OUTPUT_ROOT=$XW ../venv/bin/python poster/effect_share.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MANIFEST = (Path(os.environ.get("HBM_OUTPUT_ROOT", "/ritom/scratch/cscs/zhang/xw"))
            / "outputs/interp/manifest/contrasts-grid-en.parquet")
CONTRASTS = ("mil_combat_civ_impl", "mil_combat_civ_expl", "rel_atheist_christian_impl")


def main() -> None:
    man = pd.read_parquet(MANIFEST, columns=["contrast_id", "aligned", "dm_base", "dm_sft_v3"])
    man = man[man["aligned"] & man["contrast_id"].isin(CONTRASTS)]
    rows = {c: {"pairs": len(g),
                "removed_pct": float(100 * (1 - g["dm_sft_v3"].abs().sum()
                                            / g["dm_base"].abs().sum()))}
            for c, g in man.groupby("contrast_id")}
    if set(rows) != set(CONTRASTS):
        raise SystemExit(f"{MANIFEST}: contrasts {sorted(rows)}")
    shares = [r["removed_pct"] for r in rows.values()]
    out = {"source": "poster/effect_share.py, " + str(MANIFEST.name)
           + " (scripts/interp/04_manifest.py)",
           "contrasts": rows, "range": f"{min(shares):.0f}–{max(shares):.0f}%"}
    (HERE / "effect_share.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
