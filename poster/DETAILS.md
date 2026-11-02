# Evidence for the revised research poster

This page supports **What Changes After SFT? Auditing Hiring Decisions and Internal Pathways**.
Rebuild it with `python poster/revision_details.py` after regenerating the CPU and compute summaries.
The tables below use completed measurements, including all three registered window-r163
parameter-budget controls. The mechanism panels retain their stated selected-checkpoint scope;
additional mechanism-seed results have not been incorporated into those panels.

## Scope and interpretation

The development benchmark contains 150 candidates, each paired with three distinct jobs,
giving 450 CV–job pairs, 2,700 attribute sets and 31,050 prompts. There are 171 distinct jobs
overall, rather than three jobs shared by every CV. Each set varies identity cues while fixing qualifications, job, attribute family
and cue style. An unstable set includes both hire and reject decisions. These outcomes audit
consistency; GPT-4o agreement is a reference proxy, not independently labelled hiring quality.
Training-candidate exclusion does not make a repeatedly inspected development benchmark a clean
held-out test. The results are exploratory, English-only, on one 9B model and one shared SFT-v3
checkpoint. Post-SFT recipes have three training seeds. No held-out result is claimed.

The principal comparison fixes the hire rate at SFT-v3's 22.512%. Per-seed paired 95% intervals
resample candidates. The CPU breakdown reuses the canonical matching and tie rules, with global
cuts fixed before subgroup analysis and percentile candidate bootstrap intervals conditional on
the evaluated seeds. No three-seed aggregate confidence interval is claimed. Source hashes and tie handling
are recorded in [`reports/poster_followup/summary.json`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/summary.json) and
[`scripts/poster_audit_followup.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/scripts/poster_audit_followup.py).

## Training signals

- DPO uses 4,880 preference pairs constructed from SFT-v3's sampled mistakes and teacher-written
  chosen responses. See [`docs/V3_NEW_DATA.md`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/docs/V3_NEW_DATA.md).
- The plotted GRPO R4 runs use 4,154 tipped prompts, fixed attribute-free decision-reference
  rewards plus a format reward, and KL regularization to SFT-v3. Learning-rate decay covers
  the final 26% of the schedule. The 9,140-row released GRPO pool is not the training-row count
  of these runs. The optional operating-point-anchor method is not enabled in this recipe.
  See [`configs/mitigation/grpo_improve/r4_tipped.yaml`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/configs/mitigation/grpo_improve/r4_tipped.yaml) and
  [`src/hiring_bias_mitigation/mitigation/grpo.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/src/hiring_bias_mitigation/mitigation/grpo.py).
- Self-distillation uses frozen SFT-v3 as the teacher on the same CV–job pair without identity
  cues. It minimizes teacher-to-student KL over the two decision-token probabilities;
  attribute-free student rows are also included. See
  [`src/hiring_bias_mitigation/mitigation/decision_distill.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/src/hiring_bias_mitigation/mitigation/decision_distill.py).

## Behavioral results and operating points

At 22.51% hire rate, Base-to-SFT instability falls from 11.8% to 1.8%, a change of
−9.95 pp with paired 95% CI [−12.07, −7.83]. GPT-4o agreement rises from 78.8% to 82.4%
at each model's own greedy rate. These two comparisons use different rate conventions.

| Post-SFT recipe | Seed | Change vs SFT-v3, pp | Paired 95% CI |
| --- | --- | --- | --- |
| DPO | 42 | -0.815 | [-1.556, -0.147] |
| DPO | 43 | -0.296 | [-1.000, +0.407] |
| DPO | 44 | +0.444 | [-0.148, +1.111] |
| GRPO + lr decay | 42 | -0.315 | [-1.176, +0.472] |
| GRPO + lr decay | 43 | -0.318 | [-1.109, +0.474] |
| GRPO + lr decay | 44 | +0.137 | [-0.727, +1.004] |
| Self-distillation | 42 | -0.119 | [-0.780, +0.521] |
| Self-distillation | 43 | -0.407 | [-1.000, +0.209] |
| Self-distillation | 44 | -0.328 | [-1.038, +0.339] |

One DPO seed alone would have read as about a 1 pp gain: seed 42's own-rate comparison is
−1.03 pp, while its fixed-22.51% comparison is −0.815 pp. The poster's operating-point curves
show all 51 grid points from 10% to 35%, all three seeds, and their pointwise mean.
Fractions below SFT are fractions of these grid points, not hire-rate ranges.

| Recipe | Below SFT, seeds 42 / 43 / 44 | Seed mean below SFT |
| --- | --- | --- |
| DPO | 19/51, 20/51, 18/51 | 19/51 |
| GRPO + lr decay | 26/51, 32/51, 31/51 | 30/51 |
| Self-distillation | 34/51, 41/51, 35/51 | 39/51 |

The curve points are correlated descriptive evaluations. Their count is not a significance test.
The thin lines are individual seed trajectories, not confidence bands. Paired bootstrap
intervals are displayed only at the fixed 22.51% operating point in the adjacent forest plot.
Sources: [`reports/final/final_table.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/final/final_table.csv), [`reports/final/curves.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/final/curves.csv).
The family × cue-style breakdown is in [`reports/poster_followup/family_cue.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/family_cue.csv).
It uses global matched-rate cuts before stratification, not a new threshold for each subgroup.

## Residual instability and tail sensitivity

The poster illustrates the 43/44 seed pair at each arm's own rate. All pairs are shown here.
The control holds SFT-v3 scores fixed and recalibrates its cut to each seed's greedy hire
count, including the canonical weighted tie rule. It does not reuse the post-trained model's
numeric logit threshold. See [`scripts/seed_overlap.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/scripts/seed_overlap.py) and
[`scripts/scale_only_control.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/scripts/scale_only_control.py). A low actual overlap alone cannot separate a recipe
change from a threshold change.

| Recipe | Seeds | Actual Jaccard | SFT threshold control |
| --- | --- | --- | --- |
| DPO | 42/43 | 0.145 | 0.221 |
| DPO | 42/44 | 0.139 | 0.186 |
| DPO | 43/44 | 0.256 | 0.829 |
| GRPO + lr decay | 42/43 | 0.137 | 0.097 |
| GRPO + lr decay | 42/44 | 0.085 | 0.089 |
| GRPO + lr decay | 43/44 | 0.280 | 0.915 |
| Self-distillation | 42/43 | 0.527 | 0.829 |
| Self-distillation | 42/44 | 0.604 | 0.969 |
| Self-distillation | 43/44 | 0.453 | 0.858 |

Source: [`reports/seed_overlap/overlap.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/seed_overlap/overlap.csv). A second analysis fixes every arm at
the same 22.512% hire rate. Fractional tied-set membership uses sum(min weights)/sum(max weights).
The SFT control is then identical across seeds and equals one. Do not mix actual overlaps from
this table with the own-rate controls above.

For the illustrated seeds 43/44, changing SFT's thresholds alone leaves substantially more
overlap than the trained-seed comparison. This does not rule out every calibration or
operating-point explanation and does not establish that training creates new real-world bias.

| Recipe | Seeds | Common-rate weighted Jaccard | Same-cut SFT control |
| --- | --- | --- | --- |
| DPO | 42/43 | 0.172 | 1.000 |
| DPO | 42/44 | 0.275 | 1.000 |
| DPO | 43/44 | 0.244 | 1.000 |
| GRPO + lr decay | 42/43 | 0.327 | 1.000 |
| GRPO + lr decay | 42/44 | 0.347 | 1.000 |
| GRPO + lr decay | 43/44 | 0.299 | 1.000 |
| Self-distillation | 42/43 | 0.537 | 1.000 |
| Self-distillation | 42/44 | 0.533 | 1.000 |
| Self-distillation | 43/44 | 0.544 | 1.000 |

Source: [`reports/poster_followup/common_rate_overlap.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/common_rate_overlap.csv).
The 35 pairs near SFT's cut are 7.8% of the 450 pairs. The share of remaining instability on
these pairs depends on rate convention: the fixed-rate values are recorded per seed in
[`reports/poster_followup/common_rate_seeds.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/common_rate_seeds.csv).

The tail plot uses one selected checkpoint per method and all 450 pairs for each contrast.
Its p50 and p99 are ratios of quantiles of absolute attribute-induced margin differences,
relative to SFT-v3. The connecting line spans four contrasts; it is not a confidence interval.
At these selected checkpoints, DPO and GRPO have p50 ratios below one and p99 ratios above one
in all four contrasts. Median absolute attribute effects decrease while the extreme tail
increases. A uniform positive rescaling of all absolute effects cannot produce both changes.
These logit effects are not probabilities of discrimination or errors.

| Recipe | Contrast | Pairs | p50 ratio | p99 ratio |
| --- | --- | --- | --- | --- |
| DPO | mil_combat_civ_impl | 450 | 0.294 | 6.261 |
| DPO | mil_combat_civ_expl | 450 | 0.364 | 4.462 |
| DPO | rel_atheist_christian_impl | 450 | 0.355 | 10.689 |
| DPO | gen_male_female_impl | 450 | 0.518 | 5.507 |
| GRPO + lr decay | mil_combat_civ_impl | 450 | 0.518 | 2.211 |
| GRPO + lr decay | mil_combat_civ_expl | 450 | 0.565 | 2.467 |
| GRPO + lr decay | rel_atheist_christian_impl | 450 | 0.592 | 3.797 |
| GRPO + lr decay | gen_male_female_impl | 450 | 0.779 | 3.325 |
| Self-distillation | mil_combat_civ_impl | 450 | 0.449 | 0.404 |
| Self-distillation | mil_combat_civ_expl | 450 | 0.507 | 0.409 |
| Self-distillation | rel_atheist_christian_impl | 450 | 0.525 | 0.425 |
| Self-distillation | gen_male_female_impl | 450 | 0.643 | 0.514 |

Sources: the `sensitivity.csv` files under `reports/interp/distill/` and
[`reports/poster_followup/tail_sensitivity.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/tail_sensitivity.csv). This behavioral tail result is not
established as a consequence of the patched head changes.

## What the interventions establish

The Base-to-SFT panels select pairs by Base effect. The post-SFT comparisons select on SFT-v3
effect, with 64 pairs per contrast except gender, where 45 pairs pass selection. "Same pairs"
applies within a comparison, not across those panels. The Base-selected combat-implicit
intervention at L15.H11 removes about 0.66 logits per pair in Base and 0.13 after SFT.

Selected-pair effects do not estimate the full benchmark effect. For the implicit male/female
contrast, the Base-effect-selected subset shows about 72% reduction, while all 450 pairs show
about 25% reduction, with a paired candidate-bootstrap 95% CI of 12%–37%. The direction remains
a reduction on the full benchmark, but the estimated magnitude is sensitive to effect-based
selection. The poster displays this full-grid result prominently beside the selected subset.
The full-grid results are:

| Contrast | Pairs | Mean absolute effect removed | Candidate-bootstrap 95% CI |
| --- | --- | --- | --- |
| mil_combat_civ_impl | 450 | 51.9% | [45.3, 58.2]% |
| mil_combat_civ_expl | 450 | 65.8% | [59.3, 71.7]% |
| rel_atheist_christian_impl | 450 | 60.9% | [55.9, 65.8]% |
| gen_male_female_impl | 450 | 25.3% | [12.0, 36.7]% |

Source: [`reports/poster_followup/full_grid_effects.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/full_grid_effects.csv).
Component-profile correlations come from real mixer/MLP interventions in `real.csv`.
Head rankings first use attribution patching in `atp.csv`; `verify.csv` intervenes on a selected
subset of heads. The ranks below are among that tested subset, not all heads in the model.

| Model | Contrast | Head | Real rank | Heads tested | Logits removed / pair |
| --- | --- | --- | --- | --- | --- |
| dpo4880 | mil_combat_civ_impl | L15.H11 | 4 | 46 | 0.296 |
| dpo4880 | mil_combat_civ_impl | L23.H1 | 1 | 46 | 0.445 |
| dpo4880 | mil_combat_civ_expl | L15.H11 | 1 | 49 | 0.366 |
| dpo4880 | mil_combat_civ_expl | L23.H1 | 6 | 49 | 0.191 |
| dpo4880 | rel_atheist_christian_impl | L15.H11 | 3 | 44 | 0.531 |
| dpo4880 | rel_atheist_christian_impl | L23.H1 | 4 | 44 | 0.439 |
| dpo4880 | gen_male_female_impl | L15.H11 | 1 | 45 | 0.231 |
| dpo4880 | gen_male_female_impl | L23.H1 | 7 | 45 | 0.109 |
| grpo_r4 | mil_combat_civ_impl | L15.H11 | 1 | 49 | 0.313 |
| grpo_r4 | mil_combat_civ_impl | L23.H1 | 4 | 49 | 0.179 |
| grpo_r4 | mil_combat_civ_expl | L15.H11 | 1 | 46 | 0.459 |
| grpo_r4 | mil_combat_civ_expl | L23.H1 | 5 | 46 | 0.180 |
| grpo_r4 | rel_atheist_christian_impl | L15.H11 | 1 | 43 | 0.422 |
| grpo_r4 | rel_atheist_christian_impl | L23.H1 | 4 | 43 | 0.351 |
| grpo_r4 | gen_male_female_impl | L15.H11 | 1 | 46 | 0.398 |
| grpo_r4 | gen_male_female_impl | L23.H1 | 7 | 46 | 0.173 |
| distill43 | mil_combat_civ_impl | L15.H11 | 1 | 46 | 0.099 |
| distill43 | mil_combat_civ_impl | L23.H1 | 11 | 46 | 0.027 |
| distill43 | mil_combat_civ_expl | L15.H11 | 1 | 46 | 0.104 |
| distill43 | mil_combat_civ_expl | L23.H1 | 14 | 46 | 0.034 |
| distill43 | rel_atheist_christian_impl | L15.H11 | 2 | 45 | 0.076 |
| distill43 | rel_atheist_christian_impl | L23.H1 | 10 | 45 | 0.041 |
| distill43 | gen_male_female_impl | L15.H11 | 1 | 44 | 0.080 |
| distill43 | gen_male_female_impl | L23.H1 | 10 | 44 | 0.027 |

DPO's combat-implicit L23.H1 is first among 46 tested heads. This supports that specific
checkpoint/contrast finding, not a global later route or a complete causal circuit.
Some random-control intervals are wide, especially gender, so controls cannot all be described
as negligible. Sources: [`reports/poster_followup/head_verification.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/head_verification.csv),
[`reports/poster_followup/target_heads.csv`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/target_heads.csv).

## Completed parameter-budget control for window LoRA

The original layers 15–20 window at rank 32 has 11,403,264 trainable parameters;
the full-layer r32 adapter has 58,195,968. The registered window r163 has
58,085,376 saved parameters, 0.1900% below full, within the declared 0.2% tolerance.
All three training and dependent evaluation jobs completed successfully. Each uses the declared
final 341-step, one-epoch checkpoint, the same data and training settings as its full-layer
seed, with alpha/r fixed at two. Layer placement and rank differ, so matching parameter count
does not equalize optimization geometry, compute, training time or representational capacity.
This is self-distillation with LoRA restricted to layers 15–20, compared with full-layer LoRA.

The primary comparison is window minus same-seed full-layer instability, at the full arm's
own greedy hire count. Positive is worse. Each 95% interval uses 1,000 paired candidate
bootstrap draws with thresholds fixed. The old r32 window is recomputed on this same basis:

| Seed | Window arm | Full-arm hire rate | Change vs full, pp | Paired 95% CI |
| --- | --- | --- | --- | --- |
| 42 | Window r32, 11.4M | 23.027% | +0.430 | [-0.269, +1.097] |
| 42 | Window r163, 58.1M | 23.027% | +0.037 | [-0.722, +0.685] |
| 43 | Window r32, 11.4M | 22.831% | +0.741 | [+0.053, +1.446] |
| 43 | Window r163, 58.1M | 22.831% | +0.684 | [+0.010, +1.341] |
| 44 | Window r32, 11.4M | 22.992% | +0.624 | [-0.056, +1.259] |
| 44 | Window r163, 58.1M | 22.992% | +0.443 | [-0.095, +1.008] |

The descriptive three-seed mean gap falls from
+0.598 pp to
+0.388 pp. The matched-budget estimates
are positive for all three seeds, but only 1/3
intervals exclude zero. There is no aggregate three-seed CI and no equivalence test.
The poster's earlier +0.53 pp used a different operating-point convention; it must not be
compared directly with the new +0.39 pp. The displayed +0.60 → +0.39 pp uses one shared convention.

At the secondary common SFT-v3 hire count of 6,990/31,050, or 22.512%, the mean gaps are
+0.417 pp and
+0.339 pp. All three matched-budget intervals cross zero:

| Seed | Window arm | Change vs full at 22.512%, pp | Paired 95% CI |
| --- | --- | --- | --- |
| 42 | Window r32, 11.4M | +0.268 | [-0.312, +0.839] |
| 42 | Window r163, 58.1M | +0.063 | [-0.391, +0.490] |
| 43 | Window r32, 11.4M | +0.670 | [-0.088, +1.393] |
| 43 | Window r163, 58.1M | +0.610 | [-0.075, +1.229] |
| 44 | Window r32, 11.4M | +0.315 | [-0.345, +0.976] |
| 44 | Window r163, 58.1M | +0.345 | [-0.285, +0.895] |

The larger window adapter reduces the observed average gap. These results neither establish
equivalence with full-layer adaptation nor a general failure of localization-guided editing.
This is a registered follow-up on the exploratory development benchmark, not held-out confirmation.
The experiment is motivated by the distinction between localization and editing in Hase et al. [4].

Training, merge and reference-agreement checks:

| Seed | Arm | Last logged loss, step 340 | Merge projection retained | GPT-4o agreement | Agreement vs full, pp |
| --- | --- | --- | --- | --- | --- |
| 42 | full_r32 | 0.01243 | 92.197% | 82.548% | +0.000 |
| 42 | window_r32 | 0.02052 | 95.684% | 82.461% | -0.087 |
| 42 | window_r163 | 0.01440 | 99.069% | 82.676% | +0.129 |
| 43 | full_r32 | 0.01265 | 91.918% | 82.712% | +0.000 |
| 43 | window_r32 | 0.01857 | 95.937% | 82.654% | -0.058 |
| 43 | window_r163 | 0.01590 | 99.033% | 82.393% | -0.319 |
| 44 | full_r32 | 0.01516 | 91.999% | 82.531% | +0.000 |
| 44 | window_r32 | 0.02113 | 95.546% | 82.660% | +0.129 |
| 44 | window_r163 | 0.01787 | 99.003% | 82.773% | +0.242 |

Loss is the last logged interval at step 340 of the 341-step run, not a whole-run average.
The larger window has lower logged loss than the old window but higher loss than full layers
in each seed. All arms decide 100% of prompts, with no parse failures or refusals. Agreement
uses each model's own greedy operating point, rather than a matched rate; it remains a proxy.
The matched window's agreement differences are +0.129, −0.319 and +0.242 pp versus full.

The merge diagnostic is the projection retained along the intended adapter update:
`sum((W_after - W_before) * D) / sum(D * D)`, with `D = B @ A * scaling` computed in float32
and the merge performed in the base dtype; before/after weights are promoted to float32
to measure the change. It describes the last adapter in
the chain, not an energy fraction or a count of changed weights. The retained projection
is about 99.0% for r163, versus about 92% for full-layer r32 and 96% for old-window r32.
This difference in bf16 merge retention, alongside rank and
optimization geometry, prevents attributing the change solely to parameter count. These are
served merged-checkpoint outcomes. Source manifests, adapter settings, training-state snapshots,
quality records and hashes are indexed in [`reports/poster_followup/window_budget_guardrails.json`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/window_budget_guardrails.json).

Protocol: [`docs/POSTER_FOLLOWUP_EXPERIMENTS.md`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/docs/POSTER_FOLLOWUP_EXPERIMENTS.md). Completed source tables, parameter counts,
bootstrap specification and hashes: [`reports/poster_followup/window_budget_summary.json`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/window_budget_summary.json).
Regenerate the summary and poster macros with [`poster/latex/window_data.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/poster/latex/window_data.py).
Per-seed comparisons and completion markers are under `reports/experiment_followup/window_seed42`,
`window_seed43` and `window_seed44`; accepted jobs are in [`reports/experiment_followup/submissions.json`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/experiment_followup/submissions.json).

## NVIDIA acceleration, timing scope and cost

The platform is CSCS Alps, with four NVIDIA GH200 GPUs per node. The recorded device name is
`NVIDIA GH200 120GB`, consistent with [`docs/ACCEL_BENCH.md`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/docs/ACCEL_BENCH.md). The poster therefore omits
the earlier unsupported `96 GB HBM3` label. These logs did not measure total device memory,
so no additional physical-capacity or HBM-generation claim is inferred from the name.
The measured software stack uses vLLM 0.31, TRL 1.15 / PEFT and PyTorch DDP in bf16.
The inference benchmark has 3,105 prompts per run and 2 repeats per condition.
The three bars are per-run medians and measure generation only; startup is excluded.
With two observations, each median also equals the arithmetic mean.

| Backend | Median generation time per run | Speed relative to vLLM |
| --- | --- | --- |
| HF batch 8 | 1091 s | 16.9× |
| HF batch 32 | 470 s | 7.3× |
| vLLM | 64.7 s | 1× |

HF/vLLM decision agreement is 99.68%–99.78%. For the full 31,050-prompt generation workload,
the medians across four repeats are 793 s including startup and writeout,
and 0.105 kWh of GPU energy per run, not energy summed over the four runs.
Its 45.65 requests/s measures generation only, about 680 s, not end-to-end
throughput. It is also a per-run median. This has 4 repeats and controlled condition ordering.
Energy is the difference between NVML device total-energy counters across the complete cost
step, summed over used devices. It includes loading and writing, measures GPU device energy,
and does not measure node/CPU energy or isolate energy attributable to one process.
It replaces the old six-step GRPO timing line, which was not a rollout-only measurement.

The DDP comparison uses reviewed one-GPU repeat job 5238558 and four-GPU job 5236800,
with the same global batch 32 and 40 training steps.

| Timing and GPU-hour basis | 1 GPU | 4 GPUs | Wall speed-up | GPU-hours, 4 vs 1 |
| --- | --- | --- | --- | --- |
| Training only | 2040 s | 550 s | 3.71× | +8% |
| Whole cost step, startup included | 2271 s | 850 s | 2.67× | +50% |

The repeat ran on an exclusive node. Steps 6–40 stayed near 50 s; SM clock stayed at 1,980 MHz,
power was about 190 W of a 900 W limit, temperature 26–27°C, and throttle counters did not change.
GPU utilization was 17%–21%. Missing `causal_conv1d` and microbatch size one are possible causes
of low utilization; neither contribution was isolated. The first run's slowdown remains unexplained.
The original registration used an absolute 5% phase difference. The accepted one-sided
no-slowdown reading is post hoc and recorded in [`poster/latex/ddp_review.json`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/poster/latex/ddp_review.json).
Ledger row 4 and the full rerun account are in [`docs/ACCEL_BENCH.md`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/docs/ACCEL_BENCH.md).

Actual selected-pair metadata, rather than the requested cap, defines the 64-pair patch cost cohort.
It contains 22 measured runs, each with 64 pairs. Runtime and used-GPU-hours
are medians across those runs: 20 min and 0.33 GPU-h per model × contrast.
The 42 GiB value is the maximum of their sampled device-memory peaks, not a median peak.
Full 31,050-prompt margin scoring is a separate pass: median 32 minutes across
11 runs and maximum measured peak 87 GiB. Memory peaks are sampled NVML
device-used-memory maxima and include vLLM reservation. Timing/energy collection is documented
in [`src/hiring_bias_mitigation/utils/cost.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/src/hiring_bias_mitigation/utils/cost.py) and [`docs/COST.md`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/docs/COST.md).
Each of the patching and margin-scoring passes fits separately on one GH200 without CPU offload.
The source snapshot totals 552 **allocated** GPU-h and 144 kWh measured
GPU energy. It excludes follow-up jobs submitted after that snapshot.
All source hashes, selected cost records, excluded-pilot logic and timing scopes are in
[`reports/poster_followup/compute_summary.json`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/reports/poster_followup/compute_summary.json) and [`poster/latex/compute_data.py`](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at/blob/main/poster/latex/compute_data.py).

## References, code and data

The poster includes numbered references and printed project/data addresses. Full citations
and detailed controls are also collected here.

- [1] [Hu et al., LoRA: Low-Rank Adaptation of Large Language Models, 2021](https://arxiv.org/abs/2106.09685).
- [2] [Rafailov et al., Direct Preference Optimization, 2023](https://arxiv.org/abs/2305.18290).
- [3] [Shao et al., DeepSeekMath, introducing GRPO, 2024](https://arxiv.org/abs/2402.03300).
- [4] [Hase et al., Does Localization Inform Editing?, 2023](https://arxiv.org/abs/2301.04213).
- [5] [Zhang and Nanda, Towards Best Practices of Activation Patching, 2024](https://arxiv.org/abs/2309.16042).
- [Project code](https://github.com/Stereotypes-in-LLMs/hiring_bias_mitigation_at).
- [Hiring-bias-mitigation-v3 dataset](https://huggingface.co/datasets/Stereotypes-in-LLMs/hiring-bias-mitigation-v3).

## Reproduction

1. Run `scripts/poster_audit_followup.py` in the project environment with the canonical raw outputs.
2. Run `poster/latex/latex_data.py --compute-only` to refresh the measured compute block.
3. Run `poster/latex/plot_revision.py` to regenerate the two vector figures.
4. Run `poster/latex/window_data.py` to validate and summarize the three completed budget controls.
5. Run `poster/revision_details.py`, then compile `poster/latex/main.tex` with pdfLaTeX twice.

The poster uses vector charts and embedded fonts. The author and affiliation line remains
an explicit placeholder until the author order is agreed.
