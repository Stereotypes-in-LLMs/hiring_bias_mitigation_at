# The final table: quality and cost of every arm after SFT-v3, on one footing

Started 2026-10-29 at the user's request while preparing the final report. Branch
`xinyang/report-dpo-cost`. No new measurement: every number is recomputed from files the
standard already writes, by `scripts/final_table.py`. The decision rule stays
[`EVAL_STANDARD.md`](EVAL_STANDARD.md)'s; this page adds views beside it and changes no verdict.

**Why.** Each arm's verdict moves SFT-v3 to that arm's own hire rate. The arms sit at different
hire rates (18.0% to 23.2%), so −1.03 for DPO_V3_4880 and −1.02 for GRPO-v3-all checkpoint 525
are measured at different points and do not say the two are equally stable. The cost report on
main still dated from 2026-10-20 and had no v3 arm. Round 3's data was charged to the DPO arm
only, although GRPO-v3-all trained on the same pairs.

## Setup

- **Arms.** Every arm trained from SFT-v3 and audited under op-margin-v2, listed by hand in
  [`reports/final/arms.csv`](../reports/final/arms.csv) with its checkpoint rule (declared
  before the audit, or exploratory), seed, parent and the jobs that built its data.
  `headline = yes` marks the recipes the report compares.
- **Three quality readings**, all from the op-margin-v2 margin files (31,050 prompts, 2,700
  within-group sets):
  - **own point**: the standard's decision, SFT-v3 moved to the arm's hire rate. Recomputed with
    the same function. It reproduces every committed comparison JSON, for example DPO_V3_4880
    −1.03 [−1.81, −0.28], GRPO-525 −1.02 [−1.99, −0.12] and Distill s43 −0.59 [−1.24, −0.03].
  - **common point**: the arm moved to SFT-v3's own hire rate, **22.51%** on the margin pass.
    The user fixed this point on 2026-10-29, before anything was computed. It uses the same
    matching (`matched_comparison`, roles swapped) and the same candidate bootstrap (1,000
    draws, seed 42).
  - **curve**: unstable within-group sets at every hire rate from 10% to 35% in 0.5 pp steps,
    each arm moved there by the same matching. No interval.
- **TPR at the common point**: the hire rate on prompts GPT-4o hires, attribute-free prompts
  included as in the guardrail, for the arm moved to 22.51% (margins). The standard's greedy
  guardrails at the arm's own point are carried as they are (`reports/reference_metrics.csv`).
- **Cost** (`reports/cost/`, regenerated 2026-11-01 after the patching-informed round, from 202
  jobs: 573 steps, 355 of them measured by step records; first regenerated 2026-10-29 from 179
  jobs).
  All of it is measured GPU-h, and rows are never summed:
  - **data**: the recipe steps of the jobs that built the arm's training data. A job shared by
    several arms counts in full in each of them.
  - **train**: GPU-h to the audited checkpoint (`checkpoints.csv`), plus the parent's training
    for an arm that starts from another.
  - **extra**: data + train, i.e. one more such model on top of SFT-v3.
  - **evaluation**: reported once, since it is the same work for every arm.
  - **exploration**: every step of a method family, failed and diagnostic steps included.
  - Round 3's teacher votes and SFT-v3 samples (jobs 5115569, 5116018, 5115774, 5116328;
    12.63 GPU-h) are now shared by `dpo_v3_5k` and `grpo_v3_all` in `reports/cost/labels.csv`
    (user, 2026-10-29). Round 3's judges and score (1.28 GPU-h) stay DPO's alone.

## Result

DPO_V3_4880 now has three training seeds (`DPO_V3_4880_SEEDS.md`; seeds 43 and 44 trained and
evaluated 2026-10-29).

**In one sentence.** At SFT-v3's hire rate, DPO_V3_4880 s42 is still the only arm whose
interval stays below 0 (−0.81 [−1.56, −0.15]), but its two new seeds are not
(−0.30 [−1.00, +0.41], +0.44 [−0.15, +1.11]). No recipe's seed mean is clearly ahead:
DPO −0.22 (sd 0.63), distillation −0.28 (sd 0.15), GRPO + decay R4 −0.16 (sd 0.26). GRPO-v3-all
checkpoint 525 falls from −1.02 at its own point to −0.18 [−1.25, +0.82].

Headline recipes (`reports/final/final_table.csv`, `recipes.csv`). SFT-v3 at 22.51% has 1.81%
unstable sets, and its TPR on the margins is 57.17%.

| arm | checkpoint | own hire rate | Δ at own point, pp [95% CI] | Δ at 22.51%, pp [95% CI] | TPR Δ at 22.51% | greedy TPR Δ, own point | data + train = extra GPU-h |
|---|---|---|---|---|---|---|---|
| DPO_V3_4880 s42 | declared | 22.79% | **−1.03 [−1.81, −0.28]** | **−0.81 [−1.56, −0.15]** | +0.59 | +0.70 | 13.91 + 4.03 = 17.94 |
| DPO_V3_4880 s43 | declared | 21.59% | −0.09 [−0.88, +0.66] | −0.30 [−1.00, +0.41] | +0.51 | −2.11 | 13.91 + 4.03 = 17.94 |
| DPO_V3_4880 s44 | declared | 21.38% | +0.20 [−0.63, +1.11] | +0.44 [−0.15, +1.11] | +0.37 | −2.98 | 13.91 + 4.06 = 17.97 |
| GRPO-v3-all final | declared | 18.71% | −0.56 [−1.61, +0.41] | −0.13 [−1.10, +0.69] | −0.94 | −10.11 (fails) | 16.19 + 23.44 = 39.63 |
| GRPO-v3-all ckpt 525 | exploratory | 19.05% | **−1.02 [−1.99, −0.12]** | −0.18 [−1.25, +0.82] | −0.12 | −9.30 | 16.19 + 21.71 = 37.90 |
| GRPO+decay R3 s42 / s43 | declared | 20.09 / 21.77% | −0.71 / −0.23 | +0.10 / −0.26 | +0.67 / +0.26 | −5.03 / −2.28 | 16.19 + 23.6 ≈ 39.7 |
| GRPO+decay R4 s42 / s43 / s44 | declared | 20.9–23.2% | −0.73 / −0.44 / −0.16 | −0.31 / −0.32 / +0.14 | +0.32 / +0.12 / −0.12 | −4.70 / +1.11 / +1.00 | 16.19 + 10.8 ≈ 27.0 |
| DAPO+decay D1 s42 / s43 / s44 | declared | 20.1–22.1% | −0.83 / +0.28 / −0.22 | +0.39 / +0.43 / −0.42 | +0.66 / +1.83 / +1.33 | −5.42 / +0.47 / −4.04 | 16.19 + 4.1 ≈ 20.3 |
| Distill s42 / s43 / s44 | declared | 22.8–23.0% | −0.42 / **−0.59 [−1.24, −0.03]** / −0.70 | −0.12 / −0.41 [−1.00, +0.21] / −0.33 | −0.46 / −0.03 / −0.38 | +1.21 / +1.28 / +1.12 | 16.19 + 5.2 ≈ 21.4 |

Bold: interval below 0. All other arms, including round 2's DPO-v3, the on-policy-only DPO,
GRPO-v3, DAPO-v3-all and D2, are in `final_table.csv`, which also gives each arm's expected
fixed and broken sets at its own point.

Seed means and spreads of the recipes with more than one seed (`recipes.csv`, sd ddof 1):

| recipe | seeds | Δ own point, mean (sd) | Δ at 22.51%, mean (sd) | TPR Δ at 22.51%, mean | every seed passes the guardrails | extra GPU-h |
|---|---|---|---|---|---|---|
| DPO_V3_4880 | 3 | −0.31 (0.65) | −0.22 (0.63) | +0.49 | yes | 17.95 |
| Decision distillation | 3 | −0.57 (0.14) | −0.28 (0.15) | −0.29 | yes | 21.37 |
| GRPO + decay R4 | 3 | −0.44 (0.29) | −0.16 (0.26) | +0.11 | yes | 27.02 |
| DAPO + decay D1 | 3 | −0.26 (0.56) | +0.13 (0.48) | +1.27 | yes | 20.26 |
| GRPO + decay R3 | 2 | −0.47 (0.34) | −0.08 (0.26) | +0.46 | yes | 39.72 |

**Curves** ([`reports/final/curves.svg`](../reports/final/curves.svg), `curves.csv`; the
recipes' other seeds are the thin lines). The table below gives the mean of (arm − SFT-v3) over
the grid points in each hire-rate band, in pp, plus the share of all 51 points where the arm is
below SFT-v3 (`reports/final/curve_bands.csv`):

| arm | 10–15% | 15–20% | 20–25% | 25–30% | 30–35% | points below SFT-v3 |
|---|---|---|---|---|---|---|
| DPO_V3_4880 s42 / s43 / s44 | +0.54 / +0.65 / +0.68 | +0.21 / −0.17 / −0.24 | −0.63 / −0.42 / −0.24 | +0.09 / +0.44 / +0.07 | +1.31 / +2.55 / +2.07 | 37 / 39 / 35% |
| GRPO-v3-all ckpt 525 | +0.08 | −0.52 | −0.49 | −0.12 | +0.04 | 65% |
| GRPO+decay R3 s42 | +0.04 | −0.32 | −0.24 | −0.26 | +0.02 | 67% |
| DAPO+decay D1 s42 | −0.05 | −0.49 | −0.01 | +0.15 | −0.36 | 57% |
| Distill s42 / s43 / s44 | +0.05 / −0.08 / +0.03 | −0.30 / −0.14 / −0.20 | −0.33 / −0.41 / −0.42 | −0.14 / −0.11 / −0.19 | −0.09 / −0.12 / −0.03 | 67 / 80 / 69% |

DPO_V3_4880's margins hardly come near its cut, on every seed. 0.06%, 0.14% and 0.18% of its
prompts have |margin| < 1 nat, against 3.79% for SFT-v3, and its median |margin| is 34.3, 34.6
and 34.1 against 13.4 (snapped margins; `near_cut_pct`, `median_abs_margin` in
`final_table.csv`). Within ±0.5 pp of its own hire rate, seed 42's unstable share stays
between 0.52% and 1.11%, seed 43's between 1.43% and 1.81%, and seed 44's between 1.70% and
2.29%. SFT-v3's is between 1.73% and 2.13% in the same window around its own point.

**Cost** (`reports/final/family_cost.csv`, `eval_cost.csv`, `reports/cost/by_arm.csv`):

| | DPO after SFT-v3 | GRPO / DAPO after SFT-v3 | decision distillation |
|---|---|---|---|
| one model (extra GPU-h) | 17.95 (DPO_V3_4880, mean of 3 seeds) | 37.90 (ckpt 525), 27.0 (R4), 20.3 (D1) | 21.4 |
| of which training | 4.03–4.06 | 21.71 / 10.8 / 4.1 | 5.2 |
| exploration, every step of the family | 43.02 (27 jobs; 9.32 of it this seed round) | 238.60 (60 jobs; 8.68 failed or cut) | 69.26 (30 jobs, the 2026-11-01 round included; 23.25 before it) |

- The fixed evaluation costs 0.50 GPU-h measured per `evaluate_arm` job, median of 25 recorded
  jobs. That is 1.21 GPU-h allocated, since the job holds a 4-GPU node.
- SFT-v3 itself costs 15.40 (data) + 21.64 (training) = 37.04 GPU-h over the base model. Every
  row above sits on top of that.
- The account's total from 2026-10-01 is 488.0 GPU-h allocated (2026-11-01; 428.5 on
  2026-10-29), of which 40.5 GPU-h is idle time in recorded jobs.

## Reading

- **Fact.** At one common hire rate, DPO_V3_4880 s42 is the only arm in the table whose
  interval stays below 0, and the same recipe's seeds 43 and 44 are not. GRPO-525's gain at
  its own point mostly disappears there. At equal hire rate, every arm's TPR is within 1.9 pp
  of SFT-v3's.
- **Fact.** Across training seeds, DPO_V3_4880 spreads most (sd 0.65 pp at its own point) and
  distillation least (0.14). Distillation is the only recipe whose three seeds are all below 0
  at both points. None of its intervals at the common point excludes 0.
- **Interpretation.** No post-SFT recipe is shown to beat SFT-v3 at a fixed hire rate. The
  best-supported claim is distillation's: a small gain (about −0.3 pp at 22.51%, −0.57 at its
  own point) that repeats across seeds. DPO_V3_4880's −1.03 is one seed of three.
- **Interpretation.** GRPO-525's verdict, and its 9.3 pp TPR loss, are tied to its lower hire
  rate. The verdict compares it with SFT-v3 moved down to 19.05%, a point where SFT-v3 is less
  stable than at its own. Not confirmed as a mechanism: the curves show where each model is
  stable, not why.
- **Fact.** DPO_V3_4880's curve has the same shape on all three seeds. It is below SFT-v3 in
  the 20–25% band (−0.63, −0.42, −0.24) and above it at 10–15% and 30–35%. All three seeds
  widen margins alike. Only seed 42 has a low unstable share around its own cut. The
  distillation seeds are below SFT-v3 across most of the range, by a smaller amount.
- **Interpretation.** DPO moves margins away from the cut on every seed, which makes few sets
  straddle it. Which sets still straddle it, and so how many, varies with the seed. Distillation
  compresses the attribute's effect more evenly (`DISTILL_PATCHING.md`: a near-uniform
  turn-down), which fits its smaller seed spread. Not confirmed. The cheap check is the overlap
  of each seed's unstable sets, on the margin files already written (`DPO_V3_4880_SEEDS.md`,
  Reading).
- **Cost claims this supports.**
  - "DPO_V3_4880 trained on 4.03 GPU-h against GRPO-525's 21.71 to its checkpoint": 5.4× in
    training. The 4.04 vs 23.46 often quoted is GRPO-v3-all's whole run to step 571; checkpoint
    525 is 21.71.
  - "One more model costs 17.95 against 37.90 GPU-h, data included": 2.1×. Distillation costs
    21.37.
  - "The DPO family took 43.0 GPU-h of exploration, its seed round included; GRPO/DAPO 238.6,
    theirs included": 5.5×.
  - The often-quoted "5.8×" holds for training alone.
  - None of these is a cost per unit of stability gained: with three seeds, DPO's gain over
    SFT-v3 is not established.

## Added 2026-11-01: the patching-informed round

[`PATCHING_DISTILL.md`](PATCHING_DISTILL.md) adds eight arms to `arms.csv`: layers 15-20 for
seeds 42-44, the window with its GDN mixers (s42), rank 8 on all layers (s42) and the margin
floor (s42), all `headline = no` like the layer-restricted run before them, and decision
distillation on each of DPO_V3_4880 s42 and GRPO+decay R4 s42, `headline = yes` with one seed
each. A student's `data_jobs` list its parent's jobs and the distillation rows' jobs, each once:
for the DPO student that adds DPO's own judge and score jobs (1.28 GPU-h, extra 26.93), for the
GRPO student nothing (the same round-3 jobs). The cost tables were regenerated on 2026-11-01
with no job running (202 jobs, 355 records, 573 steps), and the Decision distillation family's
exploration now includes the round's cost arm, `patching_distill` (`families.csv`). The
window-removed ablation (section F there) is not a trained arm and is not listed.

The DPO student's standard verdict is "better" (own point, −1.50 [−2.42, −0.64], one seed, a
gain above 1 pp, so the standard asks for no second seed). As that card registers, it is not
reported as distilled DPO beating SFT-v3: most of it is its parent's seed-42 draw.

For the two students the own point and the common point disagree in direction: the DPO student
reads −1.50 against its parent's −1.03 at its own point but −0.72 against −0.81 at 22.51%, the
GRPO student −0.16 against −0.73 and −0.51 against −0.31. Their head-to-head comparisons with
the parent at the student's hire rate, −0.07 [−0.44, +0.33] and +0.02 [−0.77, +0.82], are the
readings that card registered.

## Added 2026-11-02: the distillation follow-ups

[`DISTILL_FOLLOWUP.md`](DISTILL_FOLLOWUP.md) adds five arms to `arms.csv`:
- the seed-43 students, decision distillation on DPO_V3_4880 s43 and on GRPO+decay R4 s43,
  `headline = yes` like their seed-42 counterparts, so each student recipe now has two seeds;
- two epochs of decision distillation for seeds 42-44 (`decision_distill_e2`), `headline = no`
  as a sensitivity check. Whether it replaces one epoch as the recorded arm is the user's
  decision; the card recommends against it.

The window-kept-alone adapters (section H there) are ablations, not trained arms, and are not
listed. The cost tables were regenerated on 2026-11-02 with no job running (217 jobs, 394
records, 612 steps), and the Decision distillation family's exploration now includes the
round's cost arm, `distill_followup` (`families.csv`).

With two seeds the students read, against SFT-v3 at the common point, −0.43 (DPO) and −0.40
(GRPO), and at their own points −0.73 and −0.53 (`recipes.csv`). Head to head against their own
parent seeds they are +0.25 and −0.23 (seeds 42 / 43: −0.07 / +0.58 and +0.02 / −0.49). Neither
is better than its parent by the card's rule, and neither is reported as distilled DPO or GRPO
beating SFT-v3.

## Decided 2026-11-01 (the user)

- Decision distillation stays recorded as better than SFT-v3 (the 2026-10-27 reading of rule 4,
  `DECISION_DISTILL.md`), and GRPO + decay R3 stays recorded as better. Both sit beside the
  Reading above; no generated verdict changes.
- The poster quotes the common point, 22.51%, instead of each arm's own point: decision
  distillation −0.28 pp (seeds −0.12 / −0.41 / −0.33), DPO_V3_4880 −0.22, GRPO + decay R4 −0.16
  (three-seed means, `recipes.csv`). The follow-up runs are registered in
  `DISTILL_FOLLOWUP.md`, merged into main on 2026-11-02.

## Caveats

- Three seeds per recipe at most; a seed sd from three draws is rough. GRPO-525 is exploratory
  and has one seed.
- Seed 44's margin pass and comparison were run twice: the first did not conform, because a
  file under `scripts/` was edited during the job, and the second, identical in every number,
  does (`DPO_V3_4880_SEEDS.md`, ledger row 5).
- The common point is centred on the reference arm. A second reading would give the same table
  at another fixed hire rate; the curve shows what that would look like, without intervals.
- The curves' grid points are not independent, and a band's mean is descriptive only.
- Data cost counts a shared job in full in each arm that uses it. Round 2's DPO-v3 also used
  r2em_v3, built by v1-era jobs whose cost is reconstructed and not counted here. GRPO-v3's
  round-2 samples (jobs 5108751, 5111001) sit in the DPO family's exploration, not GRPO's.
- Decision distillation's rows are built on CPU from grpo-v3-all's prompt set, so its data
  cost is GRPO-v3-all's (16.19).
- Exploration counts the steps under the family's cost arms (`reports/final/families.csv`).
  Activation patching is analysis and is left out (decision_distill 6.72, patching_distill 3.08,
  rl_patching 3.18).
- Measured GPU-h counts the GPUs a step's processes ran on. The allocation billed is higher by
  the idle share.

## Own point against the common point (read 2026-11-04)

A count over the table's 36 post-SFT rows (one per seed, every audited arm after SFT-v3), for
the poster's conclusion; computed by `poster/latex/latex_data.py` (macros `PostRuns`,
`OwnBetterRuns`, `CommonBetterRuns`, `OwnCommonSignFlips`) and checked once by hand with pandas.

- **Paired interval below 0 at the arm's own hire rate: 6 rows** (DPO_V3_4880 s42, GRPO-v3-all
  ckpt 525, Distill s43, Distill on DPO s42, Distill on GRPO R4 s43, Distill two epochs s43).
  **At the common 22.51%: 1 row** (DPO_V3_4880 s42, −0.81 [−1.56, −0.15]).
- **8 rows change sign** between the two points. Kendall's τ between the two columns is 0.52
  (0.26 over the 20 headline rows, where the counts are 5 and 1).
- The largest own-point gain, Distill on DPO s42 at −1.50, reads −0.72 at the common point;
  GRPO-v3-all ckpt 525 goes from −1.02 (own, 19.05% hire) to −0.18.

The poster states the two counts and that the operating point decides the verdict. It is a
description of this table, with no new audit, and changes no decision above.

## Quality at the common point (read 2026-11-04)

The poster's agreement with the GPT-4o reference (78.8% -> 82.4%) was read at each arm's own
hire rate, its headline 11.8% -> 1.8% at the common point; a reviewer asked for both at one
cut. `scripts/common_point_quality.py` reads agreement, TPR (hire rate where the reference
hires) and FPR (where it rejects) from the op-margin-v2 margins for the base model, SFT-v3 and
every arm of `arms.csv`, each cut to the common hire count exactly as the final table cuts
(`matched_flags`); its TPR agrees with this table's `tpr_common_pct` to 0.01 for every arm.
Output: `reports/final/common_point_quality.csv`. Descriptive, no intervals.

| Cut to 22.51% hire | Base | SFT-v3 |
|---|---|---|
| Agreement with the reference | 82.0% | 82.3% |
| TPR | 56.7% | 57.2% |
| FPR | 5.4% | 5.2% |

So the own-rate gain in agreement is the hire rate moving (the base model's margins hire
14.4%; its greedy audit 78.8% agreement is from generated decisions, which match the margin
sign on 98.8% of prompts). At the same cut SFT-v3's reference quality is similar to the base
model's, with the unstable share 11.8% -> 1.8%. Paired candidate-bootstrap intervals (1,000
draws, seed 42; `common_point_quality_ci.csv`, read 2026-11-04): Base - SFT-v3 agreement
-0.30 pp [-2.82, +2.18], TPR -0.45 pp [-7.61, +6.70]. The intervals do not exclude a 2 pp
drop either way, so "similar" is the claim, not "held" or "equivalent"; both are against the
GPT-4o reference labels, not hiring outcomes. Every post-SFT arm sits within -0.9 to +1.8 pp
of SFT-v3's agreement and TPR at the common point (DAPO+decay D1 s43 the highest on both,
GRPO-v3-all final the lowest); every arm's agreement interval against SFT-v3 holds 0.

## Where DPO's unstable sets come from at 35% hire (read 2026-11-04)

The curve shows DPO_V3_4880 about 3.5 pp above SFT-v3 at 35% hire. `scripts/rate_decomposition.py`
cuts SFT-v3 and each DPO seed to the same hire count and splits the sets: unstable under both,
fixed (unstable for SFT-v3 only) and broken (unstable for the arm only), tie bins weighted as
`matched_flags` weights them. For the broken sets it reads the spread of the set's attributed
variants (max - min of the snapped margins) and the distance of the set's mean margin from the
cut. Outputs: `reports/final/rate_decomposition.csv` and `rate_decomposition_cells.csv` (the
same by cue group and wording). GRPO+decay R4 and Distill are in the files for comparison.

| DPO_V3_4880, three seeds | at 22.51% | at 35% |
|---|---|---|
| SFT-v3 unstable sets (of 2,700) | 49 | 52 |
| DPO unstable sets | 27 / 41 / 61 | 119 / 174 / 142 |
| fixed (SFT-v3 only) | 35 / 35 / 20 | 46 / 52 / 47 |
| broken (DPO only) | 13 / 27 / 32 | 113 / 174 / 137 |
| broken sets' variant spread, median, DPO (SFT-v3 on the same sets 1.0-1.1 / 0.4-0.9) | 25 / 21 / 18 logits | 0.4 / 0.3 / 0.5 |
| broken sets' mean margin to the cut, median, DPO | 4.6 / 3.6 / 4.3 | 0.10 / 0.09 / 0.12 |
| broken weight on SFT-v3's borderline pairs | 69% / 70% / 84% | 3% / 11% / 7% |

Per seed, DPO's unstable count is SFT-v3's minus the repaired plus the new, so the curve's
gap at 35% is (new - repaired) / 2,700. The broken sets spread across all three cue groups
(gender 21-22% of the broken weight against 17% of sets, the largest excess).

**Rank scale (added 2026-11-04 after a review).** The logit-scale columns above mislead: at a
matched hire rate any monotone change of every margin (a shift, a scaling, a squeeze of one
region) leaves the selection unchanged, so "DPO's margins are larger and cluster" explains
nothing by itself, and 18-25 logits of spread is the same ordering as 2 logits if the whole
scale is stretched. `rate_decomposition_ranks.csv` reads the same sets on the rank scale: each
prompt's margin as its percentile among the arm's 31,050 prompts, a set's interval as max -
min over its cue variants (pp of prompts), and the interval centre's distance from the cut
percentile. Medians over the sets of each group, DPO_V3_4880 against SFT-v3, three seeds:

| group, three seeds | sets | rank interval, SFT-v3 | rank interval, DPO | centre to cut, SFT-v3 | centre to cut, DPO |
|---|---|---|---|---|---|
| 35%: new (DPO only) | 113-174 | 2.1-2.3 pp | 6.5-7.5 pp | 7.8-43.9 pp | 1.5-2.2 pp |
| 35%: repaired (SFT-v3 only) | 46-52 | 2.0-2.1 | 7.8-12.0 | 0.5 | 22.8-38.0 |
| 35%: stable under both | 2,474-2,535 | 2.0 | 3.1-3.9 | 24.7-25.8 | 26.5-26.8 |
| 22.51%: new | 13-32 | 1.9-2.4 | 1.7-2.0 | 2.0 | 0.4-0.5 |
| 22.51%: repaired | 20-35 | 1.8-2.0 | 1.5-2.0 | 0.5-0.6 | 1.6-2.0 |

So, on the scale that decides the selection: at 35% DPO both widens the sets' rank intervals
(the stable sets 2.0 -> 3.1-3.9 pp, the sets it breaks 2.2 -> 6.5-7.5 pp: the cue variants are
ordered further apart) and moves the sets it breaks next to its cut (1.5-2.2 pp away, from
8-44 under SFT-v3), while the sets it repairs are moved 23-38 pp away from the cut. Both
changes act together. At 22.51% the sets DPO breaks are not wider in rank than under SFT-v3
(1.7-2.0 against 1.9-2.4 pp): their 18-25 logits of spread is the stretched logit scale, and
what breaks them is their position, at the cut (0.4-0.5 pp) where SFT-v3 had them 2.0 pp away.
This is a description of the ranking two checkpoints give the same prompts, read at two cuts;
it says where the curve's reversal comes from, not why training ordered the prompts so, and
the two cuts select different sets of one model rather than two mechanisms.

## Reproducing

```bash
source scripts/cscs/env.sh
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/cost_report.py   # reports/cost/, sacct on the login node
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/final_table.py   # reports/final/, CPU, ~20 s
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/common_point_quality.py   # reports/final/common_point_quality.csv
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/rate_decomposition.py      # reports/final/rate_decomposition*.csv
$PY -m pytest tests/test_cost.py -q
```
