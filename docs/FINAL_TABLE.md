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
- **Cost** (`reports/cost/`, regenerated 2026-10-29 14:08 from 179 jobs, 292 recorded steps).
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
| exploration, every step of the family | 43.02 (27 jobs; 9.31 of it this seed round) | 238.60 (60 jobs; 8.68 failed or cut) | 23.25 (10 jobs) |

- The fixed evaluation costs 0.50 GPU-h measured per `evaluate_arm` job, median of 13 recorded
  jobs. That is 1.21 GPU-h allocated, since the job holds a 4-GPU node.
- SFT-v3 itself costs 15.40 (data) + 21.64 (training) = 37.04 GPU-h over the base model. Every
  row above sits on top of that.
- The account's total from 2026-10-01 is 428.5 GPU-h allocated, of which 30.0 GPU-h is idle
  time in recorded jobs.

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
  Activation patching is analysis and is left out (decision_distill 6.72, rl_patching 3.18).
- Measured GPU-h counts the GPUs a step's processes ran on. The allocation billed is higher by
  the idle share.

## Reproducing

```bash
source scripts/cscs/env.sh
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/cost_report.py   # reports/cost/, sacct on the login node
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/final_table.py   # reports/final/, CPU, ~20 s
$PY -m pytest tests/test_cost.py -q
```
