# Final-adapter merge comparison, completed 2026-11-02

All nine frozen checkpoints completed both scoring paths. Leaving the final
adapter unmerged reduces the estimated window-versus-full instability gap at the
common hire count. Most of that reduction comes from the full-layer adapter
becoming less stable. The window-r163 adapter's own instability changes little.
Every seed-level difference-in-differences interval includes zero, so the result
does not establish that merging caused the window's disadvantage.

The authoritative output is
[`attempts/r2-qwen-text-layout/aggregate/`](attempts/r2-qwen-text-layout/aggregate/).
The experiment uses the development benchmark; it is not held-out validation.

## Common operating point

All entries below use 6,990 hires among 31,050 prompts, or 22.512077%.
Each HF condition calibrates its own cut. No vLLM numerical threshold is reused.
The gap is window instability minus full-layer instability, in percentage
points. Positive values mean the window has more unstable attribute sets.
These are descriptive means of seeds 42, 43 and 44, without a three-seed CI.

| Window | Readout | Merged gap, pp | Unmerged gap, pp | Change in gap, pp |
|---|---|---:|---:|---:|
| r32, 11.4M parameters | Native bf16 | +0.489 | +0.332 | -0.157 |
| r32, 11.4M parameters | fp32 residual | +0.556 | +0.370 | -0.185 |
| r163, 58.1M parameters | Native bf16 | +0.395 | +0.229 | -0.165 |
| r163, 58.1M parameters | fp32 residual | +0.506 | +0.296 | -0.210 |

Source: [descriptive seed means](attempts/r2-qwen-text-layout/aggregate/descriptive_seed_means.csv).
The full-layer comparator has 58.2M parameters at rank 32. Near-matching the
parameter count does not match rank allocation, optimization geometry or layer
placement.

For r163, the paired change in the gap is:

| Seed | Native bf16 DID, pp [95% CI] | fp32 residual DID, pp [95% CI] |
|---|---:|---:|
| 42 | -0.285 [-0.620, +0.002] | -0.333 [-0.741, +0.074] |
| 43 | -0.154 [-0.382, +0.057] | -0.185 [-0.519, +0.074] |
| 44 | -0.058 [-0.274, +0.155] | -0.111 [-0.370, +0.148] |

DID is `(window - full)_unmerged - (window - full)_merged`. All 12 DID intervals,
including the r32 window and both readouts, contain or touch zero. All 24
window-versus-full intervals at the common hire count also contain or touch
zero. Intervals use 1,000 paired candidate-bootstrap draws with fixed cuts and
are conditional on the evaluated checkpoints. These intervals neither establish
equivalence nor include training-seed uncertainty. See
[gap changes](attempts/r2-qwen-text-layout/aggregate/gap_change.csv) and
[window comparisons](attempts/r2-qwen-text-layout/aggregate/window_comparisons.csv).

## Why the gap shrinks

The mean within-checkpoint change in instability, unmerged minus merged, is:

| Arm | Native bf16, pp | fp32 residual, pp |
|---|---:|---:|
| Full-layer r32 | +0.143 | +0.198 |
| Window r32 | -0.014 | +0.012 |
| Window r163 | -0.022 | -0.012 |

The reduction in the relative gap therefore gives little evidence that the
window itself improves. Full-layer instability increases in all three seeds'
point estimates under both readouts. Only one of the 18 within-checkpoint
intervals excludes zero: full-layer seed 42 under the native bf16 readout.
The result is descriptive and conditional on this operating point.

Source: [within-checkpoint changes](attempts/r2-qwen-text-layout/aggregate/within_checkpoint.csv).
The package also contains [own operating points](attempts/r2-qwen-text-layout/aggregate/own_rates.csv),
[51-point curves from 10% to 35%](attempts/r2-qwen-text-layout/aggregate/curves.csv),
and the [calibrated cuts](attempts/r2-qwen-text-layout/aggregate/cuts.csv).
Comparisons at each representation's own full-model hire count are reported
separately; those working points can differ between representations.

## What this comparison controls

Each arm uses its declared final checkpoint-341. The common SFT adapter is
merged into the base first. Only the last distillation adapter is left unmerged
in the PEFT condition. The final adapter weights match checkpoint-341 by SHA256.
No model is retrained or checkpoint selected using this comparison's results.

Both paths use identical HF inputs, batches, left padding, positions, SDPA and
base dtype. PEFT's normal adapter autocasting produces fp32 adapter weights;
the merged condition folds the update into bf16 weights. The two readouts are
reported separately: native bf16 logits use the canonical snap, while the fp32
residual readout is unsnapped. An fp32 readout does not mean the base model runs
in fp32. This tests the two serving representations, including their arithmetic,
rather than isolating weight rounding alone.

The merge jobs kept the pilot environment, where logs report PyTorch fallbacks
for both `causal_conv1d` and `chunk_gated_delta_rule`. The separate short batch
diagnostic on GPU 3 used an isolated FLA import path. Its kernel path and its
simultaneously active SFT/DPO adapters differ from this experiment. These HF
results also should not replace earlier vLLM poster numbers without stating the
changed backend, readout and batching conditions.

## Completion and verification

| Seed | Job | Final state | Whole job elapsed |
|---|---|---|---:|
| 42 | 5241383 | COMPLETED | 58m47s |
| 43 | 5241378 | COMPLETED | 59m33s |
| 44 | 5241379 | COMPLETED | 59m47s |

All 18 passes contain exactly 31,050 unique prompt IDs, 150 CVs and 450 CV/job
pairs, with 69 prompts per pair. That is 558,900 prompt scores. Prompt ordering,
tokenization identity and common scoring settings agree across every arm.
The rebuilt common-SFT text weights matched the saved SFT checkpoint bitwise.
After merging the final adapter, each rebuilt text model matched its saved
served weights bitwise. Unused vision parameters are outside these checks.

The passing pilot also loaded each saved seed-42 checkpoint directly. Both
readouts matched the rebuilt merge exactly on its 828 prompts per arm.
Independent final checks verified 594 source/metadata SHA256 values and seven
output CSV hashes. Window gaps, DID, within-checkpoint changes and seed means
agree with their defining arithmetic to within 6.7e-16 pp. A second reviewer
checked the CSV signs, scales, operating points and interval interpretation.
See [verification receipt](attempts/r2-qwen-text-layout/full_verification.json)
and [aggregate provenance](attempts/r2-qwen-text-layout/aggregate/summary.json).

No full GPU pass failed or required a retry. The initial pilot, job 5241337,
stopped before scoring because the verifier did not yet map the saved
`model.language_model.*` names to HF's text-model `model.*` names. Its files and
cost remain preserved. The reviewed mapping kept exact dtype/weight checks;
retry 5241355 passed. No numerical tolerance was relaxed. The automatic CPU
aggregate completed on the final seed node without a repair or GPU rerun.

## Measured cost

| Work | Used GPU-h | GPU energy, kWh |
|---|---:|---:|
| Nine full comparisons | 8.3802 | 4.35434 |
| Passing pilot | 0.2465 | 0.07527 |
| Initial failed pilot | 0.0886 | 0.00902 |
| Independent short batch diagnostic | 0.0287 | 0.00594 |
| Total | 8.7440 | 4.44458 |

Unique Slurm jobs consumed 12.3489 allocated GPU-hours. The three full jobs
account for 11.8744 of those hours and include the short diagnostic on seed 42's
fourth GPU. Concurrent cost steps were not summed as separate node allocations.
Used GPU-hours are whole cost-step wall time multiplied by the process-observed
GPU count. Energy comes from NVML GPU-device counters, not whole-node energy.
See [cost receipt](attempts/r2-qwen-text-layout/cost_summary.json), which records
all 16 cost files, their hashes and all five Slurm job states.

The active-source pointer is now marked complete. Aggregate CSV/JSON files and
nine-arm metadata have been copied to the local workspace; full scoring parquet
files remain on CSCS. No poster source was changed and these new full results
have not been committed by this task.
