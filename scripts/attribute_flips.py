"""Attribute-induced flips in each direction, per model: on the training pool from the
on-policy samples, and on the benchmark from the audits' greedy decisions. CPU only.

Why. Iterative DPO samples the newest model's own attribute-tipped mistakes as the next round's
negatives (generation/onpolicy.py). From r2em checkpoint-100 that found 409 tipped rows against
SFT's 5,614, nearly all with a reject reference, and most of them strongly tipped where SFT's
were noise (docs/ITERATIVE_DPO.md). The tipped count needs only one wrong sample in eight, so
it falls whenever a model samples less noise. Whether a model depends on the attribute more or
less is read from majority decisions instead:

  pool       a row's decision is the majority of its k attributed samples, its attribute-free
             decision the majority of its pair's k attribute-free samples; ties count as no
             decision. A row with no usable wrong sample is majority-right and never a flip.
  benchmark  the audit's greedy decision on each attributed prompt against the same pair's
             attr_free prompt.

    python scripts/attribute_flips.py \\
        --pool SFT=artifacts/semisynthetic-v2/onpolicy-full \\
               pi1=artifacts/semisynthetic-v2/onpolicy-it2 \\
        --audit SFT=Qwen3.5-9B--en--sft--adapter--en_only_cscs \\
                pi1=Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiring_bias_mitigation.generation.onpolicy import ROW_KEY, TIPPED  # noqa: E402
from hiring_bias_mitigation.utils.config import resolve_output_path  # noqa: E402

#: Strongly tipped: wrong in at least this share of the attributed samples, and right in at
#: least STRONG_FREE_RIGHT of the attribute-free ones (6 and 7 of 8).
STRONG_WRONG, STRONG_FREE_RIGHT = 0.75, 0.875


def pool_flips(rows: pd.DataFrame, selected: pd.DataFrame, k: int) -> dict:
    """Tipped, strongly tipped and majority-flipped rows of one on-policy run.

    `rows` is the run's pool (rows.parquet), `selected` its one-per-row negatives with
    `n_wrong_samples` and `free_right_rate` (selected.parquet), `k` its samples per prompt.
    """
    selected = selected.drop_duplicates(list(ROW_KEY))
    tipped = selected[selected["rejected_strategy"] == TIPPED]
    strong = tipped[(tipped["n_wrong_samples"] >= STRONG_WRONG * k)
                    & (tipped["free_right_rate"] >= STRONG_FREE_RIGHT)]
    out: dict = {
        "rows": len(rows),
        "tipped_rows": len(tipped),
        "tipped_mean_wrong_samples": (float(tipped["n_wrong_samples"].mean())
                                      if len(tipped) else None),
        "strong_rows": len(strong),
        "strong_pairs": int(strong["pair_id"].nunique()),
        "strong_by_reference": {str(r): int(n) for r, n in
                                strong["reference_decision"].value_counts().items()},
    }
    majority = selected[(selected["n_wrong_samples"] > k / 2)
                        & (selected["free_right_rate"] > 0.5)]
    for reference, toward in (("reject", "hire"), ("hire", "reject")):
        n_rows = int((rows["reference_decision"] == reference).sum())
        flips = majority[majority["reference_decision"] == reference]
        out[f"majority_to_{toward}"] = {
            "rows": len(flips), "pairs": int(flips["pair_id"].nunique()),
            f"{reference}_reference_rows": n_rows,
            "share": len(flips) / n_rows if n_rows else None,
        }
    return out


def benchmark_flips(audit: pd.DataFrame) -> dict:
    """Greedy decisions on attributed prompts against their pair's attr_free decision."""
    free = (audit[audit["condition"] == "attr_free"].drop_duplicates("pair_id")
            .set_index("pair_id")["decision"])
    attributed = audit[audit["condition"] != "attr_free"].assign(
        free=lambda f: f["pair_id"].map(free))
    out: dict = {"pairs": int(free.size), "attributed_prompts": len(attributed)}
    for start, toward in (("reject", "hire"), ("hire", "reject")):
        part = attributed[attributed["free"] == start]
        flipped = part["decision"] == toward
        per_pair = flipped.groupby(part["pair_id"]).mean()
        out[f"{start}_to_{toward}"] = {
            "free_pairs": int(per_pair.size),
            "prompt_share": float(flipped.mean()) if len(part) else None,
            # the reference side the flip moves away from: a wrong flip when it equals `start`
            "prompt_share_on_reference": float(
                flipped[part["reference_decision"] == start].mean()) if len(part) else None,
            "pairs_flipped_by_half": int((per_pair >= 0.5).sum()),
            "pairs_with_any_flip": int((per_pair > 0).sum()),
        }
    return out


def _labelled(entries: list[str]) -> dict[str, str]:
    pairs = [e.split("=", 1) for e in entries]
    if any(len(p) != 2 for p in pairs):
        raise SystemExit("entries are LABEL=PATH")
    return dict(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", nargs="*", default=[],
                        help="LABEL=on-policy artifacts dir (rows, selected, sample_report)")
    parser.add_argument("--audit", nargs="*", default=[],
                        help="LABEL=audit run name, read from outputs/raw/<run>.parquet")
    parser.add_argument("--out", help="also write every number as JSON here")
    args = parser.parse_args()

    report: dict = {"pool": {}, "benchmark": {}}
    for label, directory in _labelled(args.pool).items():
        root = Path(resolve_output_path(directory))
        k = json.loads((root / "sample_report.json").read_text())["samples_per_prompt"]
        report["pool"][label] = pool_flips(pd.read_parquet(root / "rows.parquet"),
                                           pd.read_parquet(root / "selected.parquet"), int(k))
    for label, run in _labelled(args.audit).items():
        path = Path(resolve_output_path("outputs/raw")) / f"{run}.parquet"
        audit = pd.read_parquet(path, columns=["pair_id", "condition", "reference_decision",
                                               "decision"])
        report["benchmark"][label] = benchmark_flips(audit)
    text = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
