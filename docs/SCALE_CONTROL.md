# Scale-only control: can a rescaling of SFT-v3's margins explain the matched gain? (registration)

The user proposed this on 2026-10-29, after DISTILL_PATCHING.md found distillation close to a
global margin compression. Branch `xinyang/interp-distill`. CPU only, from the op-margin-v2
margin files the standard already reads (`$HBM_OUTPUT_ROOT/outputs/operating_point/v2/`); no
model is run.

## Why

The standard compares an arm at its greedy threshold with SFT-v3 moved to the same hire count
(`operating_point_control.matched_comparison`). Suppose an arm's margins were only
m_arm = a · m_SFT-v3 with a > 0. Then:
- its zero-threshold decisions are SFT-v3's;
- re-cut at any hire count, its ranking is SFT-v3's.

So a pure rescaling, compressing (a < 1) or widening (a > 1), cannot by itself change unstable
sets at matched hire rate. Two things can still enter.
- **Precision and ties.** Every arm's margins near the threshold sit on a 1/8-logit grid
  (differences of bf16 logits). Within ±2 logits there are 33 distinct values per arm. The
  matching hires a tied bin whole sets at a time, so tied variants of one set are never split.
  Compressing margins onto a fixed grid merges neighbours into ties: SFT-v3 has 2,354 rows
  within ±2 logits, distillation s43 4,486, DPO_V3_4880 44 and GRPO-525 201. Merged ties can
  hide flips mechanically.
- **What the arm does beyond the rescaling.** That is the residual against a scaled SFT-v3.

## Arms

| arm | run | standard's matched change vs SFT-v3 |
|---|---|---|
| distillation s42 / s43 / s44 | `…en_only_decision_distill_seed{42,43,44}` | −0.42 / −0.59 / −0.70 |
| DPO_V3_4880 | `…dpo--adapter--en_only_dpo_v3_4880_step305` | −1.03 |
| GRPO-525 | `…grpo--adapter--en_only_grpo_v3_all_step525` | −1.02 |

## Baselines, built from SFT-v3's own margins (prompt for prompt)

- **S_exact(a)**: a · m_SFT-v3, no requantization.
- **S_q(a)**: a · m_SFT-v3, rounded to the 1/8 grid that the arms' margins sit on. This is a
  model that compresses or widens exactly as a, read through the same precision.
- **a per arm**: two values.
  - The median attribute-free |m| ratio, the same definition as r_scale in
    DISTILL_PATCHING.md, computed on these files.
  - The least-squares slope through the origin over all prompts.
- **A sweep**: a ∈ {0.25, 0.5, 0.75, 1, 1.5, 2, 3}, for S_exact and S_q.

## Measures

All through `matched_comparison` unchanged, with its 1,000-draw candidate bootstrap.
- (1) S_exact(a) and S_q(a) against SFT-v3.
- (2) Each arm against S_q(a), with a the arm's own value: the matched change beyond its
  rescaling.
- (3) Where the arm's matched change sits.
  - Of the sets fixed and broken against SFT-v3 at matched hire rate, the share on borderline
    pairs. Borderline means SFT-v3's attribute-free |m| < 2; the pairs are counted on these
    files.
  - Within those sets, whether the order of the variants' margins changed: Spearman between
    the arm's margins and SFT-v3's inside each set, against the same correlation for S_q.

## Predictions

- **P1, the sanity check.** S_exact(a) gives a matched change of exactly 0.000 and identical
  set flags for every a. Anything else means the comparison is not scale-invariant, and that
  is reported as a finding about the metric before anything else.
- **P2, precision.** S_q(a) for a < 1 gives a change ≤ 0; for a > 1 a change near 0.
  - If S_q at distillation's own a reaches half or more of distillation's change (≤ −0.30 pp
    for s43), a substantial part of the distillation result is a precision-and-tie artifact.
    That gets flagged in DECISION_DISTILL.md.
- **P3, the residual.** Each arm against S_q(own a) stays below 0, at a size near its
  standard change. Otherwise the arm's gain is mostly its rescaling read through the grid.
- **P4, where.** At least half of distillation's fixed sets lie on borderline pairs (prior
  from DECISION_DISTILL.md section 5b), and those sets show a lower within-set rank
  correlation with SFT-v3 than S_q gives them, i.e. a local reordering.
  - For DPO and GRPO there is no prior; the result is reported as found.

## Scope

- No change to the standard, its grid or its tie rule. This reads it, it does not amend it.
- The DPO/GRPO reading "they widen the margins" stays a prediction in RL_PATCHING.md, whatever
  this finds: P1 is exactly the reason a widening alone cannot be the explanation.
- Code: `scripts/scale_only_control.py`, with a CPU test. Outputs:
  `reports/scale_control/{summary.csv,sweep.csv,attribution.csv}`.
