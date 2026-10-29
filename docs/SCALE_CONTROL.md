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

## Result (2026-10-29)

`python scripts/scale_only_control.py` at commit 67e576f, about 15 s on CPU. Files:
`reports/scale_control/{sweep,summary,attribution}.csv` (registered) and
`posthoc_spread_center.csv`, `posthoc_decomposition.csv` (post hoc).

**Counts.** The registration's counts of rows within ±2 logits (2,354 / 4,486 / 44 / 201) used
raw margins. The script snaps first, which drops values exactly at ±2. On the script's
counting: 2,309 (SFT-v3), 4,311 (distillation s43), 43 (DPO), 191 (GRPO).

**Scale factors on these files.**

| arm | median attribute-free \|m\| ratio | slope through the origin |
|---|---|---|
| distillation s42 / s43 / s44 | 0.466 / 0.495 / 0.515 | 0.525 / 0.556 / 0.577 |
| DPO_V3_4880 | 2.659 | 2.144 |
| GRPO-525 | 2.189 | 1.724 |

DISTILL_PATCHING.md's r_scale (0.464 for s43) is the same statistic on HuggingFace fp32
margins; these are vLLM's.

**P1, sanity: holds.** S_exact(a) gives a matched change of exactly 0.000 [0.00, 0.00] for every
a in the sweep and at each arm's two values of a. Its set flags at the greedy threshold are
identical to SFT-v3's. The comparison is scale-invariant.

**P2, precision: no effect on the matched change.**
- Read through the 1/8 grid, the rescaling changes greedy decisions at a ≤ 0.5. For example,
  at a = 0.25 the hire rate goes 22.51% → 22.05% and unstable sets 1.81% → 1.74%.
- But the matched change is still exactly 0.000 at every a. A requantized rescaling is still
  monotone in SFT-v3's margins, so its greedy cut falls on one of SFT-v3's bin boundaries, and
  the matched reference hires exactly the same prompts.
- The merged ties do not reach the standard's statistic. Nothing is flagged in
  DECISION_DISTILL.md.

**P3, residual: holds.** Each arm against its own S_q(a_median):

| arm | vs SFT-v3 | vs S_q(a_median) |
|---|---|---|
| distillation s42 | −0.42 [−1.22, +0.38] | −0.46 [−1.26, +0.33] |
| distillation s43 | −0.59 [−1.24, −0.03] | −0.67 [−1.31, −0.11] |
| distillation s44 | −0.70 [−1.52, +0.08] | −0.54 [−1.30, +0.19] |
| DPO_V3_4880 | −1.03 [−1.81, −0.28] | −1.03 [−1.81, −0.28] |
| GRPO-525 | −1.02 [−1.99, −0.12] | −1.02 [−1.99, −0.12] |

The two columns differ only through the tie weights at the matched cut. None of the arms' gains
is their rescaling.

**P4, where: holds in its first part; its second part does not single out the moved sets.**
- 35 pairs are borderline (SFT-v3's attribute-free |m| < 2), i.e. 210 of 2,700 sets (7.8%).
- Distillation: 83–87% of the fixed sets lie on them, and 100% of the broken sets (s42 / s43 /
  s44: 32.8 / 32.1 / 38.9 fixed, 21.5 / 16.0 / 20.1 broken, weighted at the tied cut).
- DPO: 88% of the fixed sets and 69% of the broken. GRPO: 49% and 43%.
  - GRPO hires 19%, so its matched cut sits away from SFT-v3's zero, and "borderline at 0"
    is the wrong band for it.
- Within-set rank correlation with SFT-v3, in the sets that moved: arm 0.76–0.79 against
  S_q's 0.94–0.96 for distillation. But across all sets the arm sits at 0.52–0.54 against
  0.87–0.88. Distillation reorders variants everywhere, and less in the sets that moved, so a
  within-set reordering is not what distinguishes them.

**Post hoc: what moves the sets.** Neither check was registered.
- *Ratios in the moved sets* (`posthoc_spread_center.csv`). In distillation's fixed sets, the
  within-set range is 0.81–0.87 of a times SFT-v3's, and the set median's distance to the cut
  is 2.3–3.2 times a times SFT-v3's. In the broken sets the spread is 0.97–1.02 and the
  distance 0.00–0.28. These are conditioned on the outcome: a fixed set must end up away from
  the cut. So they describe the moved sets, they do not cause anything.
- *Hybrids* (`posthoc_decomposition.csv`), each through the standard matching against SFT-v3.
  "Centers only" takes the arm's set medians with the rescaled SFT-v3's within-set deviations;
  "deviations only" takes the reverse.

| arm | arm | centers only | deviations only |
|---|---|---|---|
| distillation s42 | −0.42 [−1.22, +0.38] | −0.30 [−1.07, +0.40] | −0.01 [−0.29, +0.29] |
| distillation s43 | −0.59 [−1.24, −0.03] | −0.38 [−1.16, +0.40] | −0.03 [−0.33, +0.27] |
| distillation s44 | −0.70 [−1.52, +0.08] | +0.02 [−0.81, +0.85] | −0.21 [−0.47, +0.01] |
| DPO_V3_4880 | −1.03 [−1.81, −0.28] | −1.99 [−2.88, −1.19] | +3.52 [+2.29, +4.77] |
| GRPO-525 | −1.02 [−1.99, −0.12] | −1.64 [−2.78, −0.71] | +1.19 [+0.21, +2.29] |

## Reading

- **Fact.**
  - A pure rescaling, exact or through the 1/8 grid, leaves the standard's statistic at
    exactly 0. Every arm's gain survives against its own rescaled SFT-v3.
  - Distillation's moved sets sit almost entirely on the 35 borderline pairs.
  - In the hybrids, the arm's set centers carry most of the effect, more than its within-set
    deviations.
    - Distillation: in two seeds out of three; the parts do not add, s44 −0.19 against −0.70.
    - DPO and GRPO: their centers alone would give about twice their gain, and their own
      deviations cost much of it back.
- **Interpretation, not confirmed.**
  - What relates to the decision change is where each pair as a whole lands relative to the
    cut, beyond the rescaling. That is a re-ranking of borderline pairs, not attribute
    invariance inside a set.
  - For distillation this is modest and seed-dependent.
  - For DPO and GRPO it is large and partly offset by their heavy-tailed attribute effects
    (RL_PATCHING.md). This is consistent with the per-pair effects there: median |Δm| down,
    upper tail up.
- **What this does not support.**
  - That "margins got smaller" or "margins got larger" explains any arm's gain. P1 rules that
    out by construction.
  - The hybrids are not models. They re-cut at their own hire rates, and their parts interact.
- **Unknown.**
  - Why a pair's center moves: the attribute-free anchor in distillation, or the reward's
    reference verdict in GRPO, are candidates; neither is tested.
  - One seed each for DPO and GRPO.

## Caveats

- These are the op-margin-v2 margins (vLLM, the standard's own), not the HuggingFace fp32
  margins of the patching cards. The a values differ slightly between the two (s43: 0.495
  against 0.464).
- Borderline is defined at SFT-v3's zero threshold. That is right for the arms that keep the
  hire rate (distillation, DPO) and not for GRPO.
- Rank correlations inside a set of 5 to 20 variants are coarse, and many sets' variants sit
  within a few grid steps of each other.
