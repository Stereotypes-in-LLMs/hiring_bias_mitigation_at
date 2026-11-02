"""CPU comparisons of the frozen window-r163 adapter with the two r32 arms.

Primary delta: window minus full at the full arm's own hire count. Secondary:
all three arms at SFT-v3's own hire count. Positive instability delta is worse.
Writes one independent per-seed report, so concurrent evaluations do not share
an output file. Missing scores fail; this script never starts model scoring.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from poster_audit_followup import (REF, ROOT, digest, load_matching, load_scores,
                                   paired_bootstrap, score_paths)


def require_same_prompts(reference, compared, opc):
    left = pd.MultiIndex.from_frame(reference[opc.KEY])
    right = pd.MultiIndex.from_frame(compared[opc.KEY])
    if left.has_duplicates or right.has_duplicates or len(left) != len(right) or len(left.difference(right)):
        raise ValueError("compared arms must contain exactly the same unique prompt keys")


def full_rate_comparison(full, window, opc, n_boot, seed):
    """Invert the canonical full-minus-window result at full's own hire count."""
    require_same_prompts(full, window, opc)
    raw = opc.matched_comparison(window, full, n_boot=n_boot, seed=seed)
    return {"hire_rate_pct": 100 * raw["hire_rate_run"],
            "unstable_full_pct": 100 * raw["unstable_run"],
            "unstable_window_pct": 100 * raw["unstable_ref_matched"],
            "delta_window_minus_full_pp": -raw["delta_matched_pp"],
            "ci_low": -raw["ci_high"], "ci_high": -raw["ci_low"],
            "window_threshold": raw["ref_threshold_matched"],
            "window_tie_weight": raw["ref_tie_weight"],
            "prompts": raw["prompts"], "sets": raw["sets"]}


def common_rate_comparison(full, arm, hire_count, opc, n_boot, seed):
    require_same_prompts(full, arm, opc)
    full_flags, full_cut = opc.matched_flags(full, "margin", hire_count)
    flags, cut = opc.matched_flags(arm, "margin", hire_count)
    flags = flags.reindex(full_flags.index)
    if flags.isna().any():
        raise ValueError("compared arms have different counterfactual sets")
    delta = flags - full_flags
    lo, hi = paired_bootstrap(delta, n_boot, seed)
    return {"hire_rate_pct": 100 * cut["hire_rate"], "hire_count": hire_count,
            "unstable_full_pct": 100 * full_flags.mean(),
            "unstable_arm_pct": 100 * flags.mean(),
            "delta_arm_minus_full_pp": 100 * delta.mean(), "ci_low": lo, "ci_high": hi,
            "arm_threshold": cut["threshold"], "arm_tie_weight": cut["tie_weight"],
            "full_threshold": full_cut["threshold"], "full_tie_weight": full_cut["tie_weight"],
            "prompts": len(arm), "sets": len(flags)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=(42, 43, 44))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()
    input_root = Path(os.environ["HBM_OUTPUT_ROOT"])
    opc = load_matching()
    runs = {"full_r32": f"Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed{args.seed}",
            "window_r32": f"Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_l15_20_seed{args.seed}",
            "window_r163": f"Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_l15_20_r163_seed{args.seed}"}
    frames, metadata = {}, {}
    sft, sft_meta = load_scores(input_root, REF, opc)
    for arm, run in runs.items():
        frames[arm], metadata[arm] = load_scores(input_root, run, opc)
        opc.check_comparable(sft_meta, metadata[arm])
        require_same_prompts(sft, frames[arm], opc)
    hire_count = int((opc.snap(sft["margin"]) > 0).sum())
    primary = [{"arm": arm, **full_rate_comparison(frames["full_r32"], frames[arm],
                                                  opc, args.n_boot, 42)}
               for arm in ("window_r32", "window_r163")]
    common = [{"arm": arm, **common_rate_comparison(frames["full_r32"], frame,
                                                    hire_count, opc, args.n_boot, 42)}
              for arm, frame in frames.items()]
    out = ROOT / "reports/experiment_followup" / f"window_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(primary).to_csv(out / "full_rate_comparison.csv", index=False)
    pd.DataFrame(common).to_csv(out / "common_rate_comparison.csv", index=False)
    record = {"training_seed": args.seed, "standard": "op-margin-v2",
              "primary_sign": "window minus full; positive is worse",
              "primary_operating_point": "same-seed full-layer r32 arm's own greedy hire count",
              "secondary_operating_point": "SFT-v3's own greedy hire count, equal for all arms",
              "bootstrap": {"draws": args.n_boot, "seed": 42, "unit": "candidate", "cuts": "fixed"},
              "canonical_source_sha256": opc.source_sha256,
              "score_sha256": {run: digest(score_paths(input_root, run)[0])
                               for run in (REF, *runs.values())},
              "inputs": {"sft_v3": sft_meta, **metadata}, "primary": primary, "common": common}
    (out / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    print(pd.DataFrame(primary).to_string(index=False))
    print(pd.DataFrame(common).to_string(index=False))


if __name__ == "__main__":
    main()
