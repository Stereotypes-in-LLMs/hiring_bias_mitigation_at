# DPO_V3_4880: two more training seeds

Started 2026-10-29 at the user's request, while preparing the final report: add training seeds
to DPO_V3_4880, the run whose result is the cleanest and cheapest so far ("补 43、44，共 3 个").
Branch `xinyang/report-dpo-cost` (from main d2c4fe0). The comparison is
[`EVAL_STANDARD.md`](EVAL_STANDARD.md), unchanged; the reference arm is SFT-v3.

**Outcome (2026-10-29): Against.** Seed 43 −0.09 pp [−0.88, +0.66], seed 44 +0.20
[−0.63, +1.11]; every guardrail passes on both. Seed 42's −1.03 did not replicate. Three-seed
mean −0.31 pp (sd 0.65); see [Result](#result).

**Seed 42, the run being replicated** (`docs/V3_NEW_DATA.md`, round 3; job 5118984 training,
5119349 evaluation). Against SFT-v3 at matched hire rate: −1.03 pp [−1.81, −0.28], every
guardrail passes, step 305 declared before training
(`reports/operating_point/v2/Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305--vs--Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional.json`:
`delta_matched_pp` −1.0349, `ci_low` −1.8133, `ci_high` −0.2825; hire rate 22.79% against
SFT-v3's 22.51% on the margin pass).

**Two kinds of uncertainty.** The interval of each comparison resamples candidates: it is the
uncertainty that comes from the 450 benchmark pairs, for one trained model. A new training seed
draws a new model from the same recipe (LoRA initialisation, data order). More bootstrap draws
cannot stand in for it. This round measures the second; each seed's own interval is still the
first.

## Pre-registration

Written on 2026-10-29 before either training job was submitted.

- **Question.** Is DPO_V3_4880's gain over SFT-v3 a property of the recipe, or of seed 42?
- **Hypothesis.** The recipe gives a gain: seeds 43 and 44 also land below SFT-v3.
- **Control.** SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`;
  op-margin-v2 `compare-arms` on 31,050 prompts and 2,700 within-group sets, via
  `scripts/cscs/evaluate_arm.sbatch`, as for seed 42.
- **What is fixed.** Everything except `seed`, `output_dir` and `wandb_run_name`
  (`configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v3_4880_seed{43,44}_gh200.yaml`, a `diff`
  against the seed-42 config shows only those three lines):
  - data `configs/data/en_only_dpo_v3_5k.yaml`, `data_kind` dpo_v3_5k, the same 4,880 training
    and 350 validation pairs;
  - start `outputs/sft/qwen3.5-9b_en_only_sft_v3_additional`;
  - beta 0.3, sigmoid + 0.2 NLL, lr 5e-5, 10 warmup steps, effective batch 16, one epoch
    (305 steps), `max_seq_len` 3072, LoRA r 32 / α 64.

  The seed also picks the seeded 200 of the 350 validation pairs that `eval_loss` reads; that
  readout selects nothing here.
- **Checkpoint.** The final one, step 305, for each seed, declared now
  (`configs/audit/qwen3.5-9b_en_dpo_v3_4880_seed{43,44}_step305_gh200.yaml`). No other
  checkpoint of these runs is audited for the result.
- **For.** Both new seeds have Δ < 0 and pass every guardrail. Then the three seeds have the
  same sign, and the standard's rule 4 is met however small each gain is.
- **Against.** Either new seed has Δ ≥ 0 or fails a guardrail. If both do, seed 42's verdict is
  reported as one seed that did not replicate, and the report's DPO claim is weakened to that.
- **Read beside it, not deciding.** The three seeds' mean and spread, against decision
  distillation (three seeds, mean −0.57 pp, `DECISION_DISTILL.md`) and GRPO + decay (five draws,
  mean −0.45 pp, spread ~0.4 pp, `GRPO_IMPROVE.md`, round 5). Hire rate and TPR against SFT-v3
  for each seed, since seed 42 kept SFT-v3's operating point.
- **Stop conditions.** `memory_floor_gb` 12; a 2 h limit per training job with `resume: true`;
  a cap of 12 GPU-h for the round (registered estimate below). One Slurm job at a time on the
  account: training 43, its evaluation, training 44, its evaluation, each submitted after the
  previous one finished and was checked.
- **Splits.** No model selection. The benchmark audit is the final test, spent once per seed.
- **Cost.** From seed 42's records: 4.04 GPU-h training (job 5118984, 1:01 on 4 GPUs) and an
  evaluation job of 15 min on 4 GPUs (job 5119349: 1.0 GPU-h allocated, the audit step 0.23
  measured). About 10 GPU-h allocated for two seeds.
  Costs under `HBM_COST_ARM=dpo_v3_4880_seeds`, so that the round-3 arm (`dpo_v3_5k`) keeps
  what producing the recipe once took.

## Ledger

GPU-h are measured steps from `reports/cost/steps.csv`; allocated hours in brackets from
`reports/cost/jobs.csv`.

| # | Step | Job(s) | GPU-h | State |
|---|---|---|---|---|
| 1 | seed 43: training | 5198797 (commit 1912a3d) | 4.04 [4.06] | done 2026-10-29, 1:00:54, step 305 saved |
| 2 | seed 43: evaluation | 5199636 (commit 1912a3d) | 0.50 [1.21] | done 2026-10-29, 0:18:09 |
| 3 | seed 44: training | 5199738 (commit 1b981f6) | 4.06 [4.09] | done 2026-10-29, 1:01:18, step 305 saved |
| 4 | seed 44: evaluation | 5200680 (commit 1b981f6) | 0.50 [1.22] | done 2026-10-29, 0:18:19; comparison did not conform (row 5) |
| 5 | seed 44: margin pass and comparison again | 5200964 (commit d047eaa), `op_control.sbatch` | 0.21 [0.85] | done 2026-10-29, 0:12:45; conforms |

Spent: 9.31 GPU-h measured, 11.43 allocated, of the 12 GPU-h cap.

Row 5 was not planned. While job 5200680 ran, `scripts/final_table.py` was edited in the
worktree the job ran from, and the standard stamps any uncommitted file under `scripts/`, so
the margin file and the comparison did not conform (verdict "undecided"). That file is not on
the scoring path. The edit was committed (d047eaa), the non-conforming files were moved to
`$XW/logs/nonconforming_5200680/`, and the margin pass and comparison were run again with no
other change. The comparison came out identical to the last digit. Single margins differ by
at most 0.0625 nat between the two passes (vLLM numerics), and the matched cut is unaffected.
The greedy audit of job 5200680 (`eval/results/…seed44_step305.json`) is kept: its record
carries no commit stamp.

## Result

**In one sentence.** The registered Against rule holds. Seed 44 is above SFT-v3
(+0.20 pp [−0.63, +1.11]) and seed 43 is level with it (−0.09 [−0.88, +0.66]), so seed 42's
−1.03 did not replicate. Interpretation: the recipe's effect is closer to −0.3 pp, with a seed
spread of about 0.65 pp. Unknown: what makes seed 42 different, since three seeds cannot place
it in a distribution.

Against SFT-v3 at matched hire rate, all on op-margin-v2, 31,050 prompts and 2,700
within-group sets. Own point from `reports/reference_metrics.csv` and
`reports/operating_point/v2/<run>--vs--Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional.json`;
the rest from `reports/final/final_table.csv` (`scripts/final_table.py`).

| Seed | Run | Hire rate | Δ own point, pp [95% CI] | fixed / broken sets | Δ at SFT-v3's 22.51%, pp [95% CI] | greedy TPR vs GPT-4o, Δ pp | agreement Δ pp | merge kept | guardrails | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | `…en_only_dpo_v3_4880_step305` | 22.79% | **−1.03 [−1.81, −0.28]** | 40.5 / 12.5 | **−0.81 [−1.56, −0.15]** | +0.70 | +0.42 | 95.85% | 4/4 pass | better |
| 43 | `…en_only_dpo_v3_4880_seed43_step305` | 21.59% | −0.09 [−0.88, +0.66] | 28.7 / 26.3 | −0.30 [−1.00, +0.41] | −2.11 | −0.26 | 95.57% | 4/4 pass | not better: the interval reaches 0 |
| 44 | `…en_only_dpo_v3_4880_seed44_step305` | 21.38% | +0.20 [−0.63, +1.11] | 31.5 / 37.0 | +0.44 [−0.15, +1.11] | −2.98 | −0.65 | 95.83% | 4/4 pass | not better: the interval reaches 0 |

Bold: interval below 0. Fixed / broken are expected counts of the 2,700 sets (the tied bin
at SFT-v3's matched cut is weighted, so they are not whole); (broken − fixed) / 2,700 is the
own-point Δ. Each served model is `outputs/merged/qwen3.5-9b_en_only_dpo_v3_4880[_seedNN]--checkpoint-305`.

**Mean and spread, read beside it** (`reports/final/recipes.csv`; sd across seeds, ddof 1):

| recipe | seeds | Δ own point, mean (sd) | Δ at 22.51%, mean (sd) | extra GPU-h per model |
|---|---|---|---|---|
| DPO_V3_4880 | 3 | −0.31 (0.65) | −0.22 (0.63) | 17.95 |
| Decision distillation | 3 | −0.57 (0.14) | −0.28 (0.15) | 21.37 |
| GRPO + decay R4 | 3 | −0.44 (0.29) | −0.16 (0.26) | 27.02 |
| DAPO + decay D1 | 3 | −0.26 (0.56) | +0.13 (0.48) | 20.26 |

The registered comparison for GRPO + decay was five draws with mean −0.45 pp
(`GRPO_IMPROVE.md`, round 5); the R4 row above is the three-seed recipe within it.

**Hire rate and TPR.** Seed 42 kept SFT-v3's operating point (22.79% against 22.51%); seeds 43
and 44 hire less (21.59%, 21.38%), and their greedy TPR on prompts GPT-4o hires is 2.1 and 3.0
pp below SFT-v3's, inside the 10 pp guardrail. At the common hire rate their TPR is
+0.51 and +0.37 pp against SFT-v3 (margins), so the greedy drop goes with the lower hire rate.

**Margins and curves** (`final_table.csv`, `reports/final/curve_bands.csv`,
`reports/final/curves.svg`, where seeds 43 and 44 are the thin blue lines):

- All three seeds push margins away from 0 alike: 0.06%, 0.14% and 0.18% of prompts are within
  1 nat of the cut, against 3.79% for SFT-v3, and the median |margin| is 34.3, 34.6 and 34.1
  against 13.4.
- The low instability around their own cut does not repeat. Within 0.5 pp of its own hire
  rate, seed 42's unstable share stays between 0.52% and 1.11%, seed 43's between 1.43% and
  1.81%, seed 44's between 1.70% and 2.29%; SFT-v3's is 1.73% to 2.13%.
- The shape of the curve repeats. Every seed is below SFT-v3 in the 20–25% band (−0.63,
  −0.42, −0.24 pp on average over its grid points) and above it at 10–15% (+0.54, +0.65,
  +0.68) and 30–35% (+1.31, +2.55, +2.07). Each is below SFT-v3 at 35–39% of the 51 grid points.

**Cost per seed.** Training 4.04, 4.04 and 4.06 GPU-h to step 305 (`checkpoints.csv`); data
13.91 GPU-h, built once and shared by all three; evaluation 0.50 GPU-h measured, 1.2 allocated.

**Raw cases.** None quoted. The question is a seed-level aggregate; which sets each seed fixes
and breaks is not yet scripted (see Reading).

## Reading

- **Fact.** Three seeds of one recipe, with data, start, config and checkpoint rule fixed, give
  own-point Δ of −1.03, −0.09 and +0.20 pp. The spread between seeds (sd 0.65) is as wide as
  one seed's bootstrap interval (half-width about 0.8). Only seed 42 is better than SFT-v3.
- **Interpretation.** Seed 42's verdict looks like a favourable draw from a recipe whose mean
  is about −0.3 pp, no better than decision distillation's −0.57 and with about 4.5 times its
  seed spread. Not confirmed: three seeds give a mean and a spread, not a distribution.
- **Fact.** What DPO does to the margins (widening them away from 0) repeats on every seed; the
  low instability at seed 42's own cut does not.
- **Interpretation.** Wide margins make few sets straddle the cut, but which sets still
  straddle it depends on the seed. This is one explanation. Another is that seed 42's cut
  happens to fall in a sparse part of its margin distribution.
- **Cheapest checks that tell them apart**, CPU only, minutes, on the margin files already
  written:
  - Overlap of the sets each seed leaves unstable at its own point. Shared sets point to the
    data, different sets to the seed.
  - Each seed's unstable share against hire rate at 0.1 pp steps around its cut. A narrow dip
    for seed 42 alone points to the cut's position.
- **What this supports for the report.** DPO_V3_4880 is the cheapest recipe per model
  (17.95 GPU-h over SFT-v3) and keeps every guardrail on all three seeds. It is not shown to
  reduce instability more than SFT-v3; seed 42's −1.03 should be reported as one seed of three.

## Caveats

- Three seeds; their sd is a rough figure.
- The seed also changes the 200 of 350 validation pairs `eval_loss` reads; that readout
  selected nothing here.
- Row 5 of the ledger re-ran a margin pass after its verdict was known. It changed no number,
  and its only purpose was the conformance stamp.
- Seed 42 was trained at c47d156 (job 5118984), seeds 43 and 44 at 1912a3d and 1b981f6. Between
  c47d156 and 1b981f6, `mitigation/dpo.py`, `scripts/cscs/train_dpo.sbatch`,
  `configs/data/en_only_dpo_v3_5k.yaml` and the seed-42 config are unchanged.
  `training_common.py` gained `lora_layer_range` and `lr_scheduler_kwargs`, both used only
  when a config sets them, and these configs do not. The cost records of the three jobs show
  the same software: torch 2.13.0, transformers 5.17.0, trl 1.15.0, peft 0.21.2.

## Reproducing

```bash
source scripts/cscs/env.sh
for s in 43 44; do   # one job at a time on the shared account
  HBM_COST_ARM=dpo_v3_4880_seeds REPO=$PWD \
    CONFIG=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v3_4880_seed${s}_gh200.yaml \
    scripts/cscs/submit.sh -p normal -t 02:00:00 scripts/cscs/train_dpo.sbatch
  HBM_COST_ARM=dpo_v3_4880_seeds REPO=$PWD \
    AUDIT=configs/audit/qwen3.5-9b_en_dpo_v3_4880_seed${s}_step305_gh200.yaml \
    AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
    scripts/cscs/submit.sh scripts/cscs/evaluate_arm.sbatch
done
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/cost_report.py
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/final_table.py
```

W&B runs `dpo-qwen3.5-9b_en_only_dpo_v3_4880_seed43` (3e07aebc) and `…_seed44` (87fb8c30).
