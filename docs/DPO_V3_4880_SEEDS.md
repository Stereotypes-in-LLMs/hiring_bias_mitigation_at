# DPO_V3_4880: two more training seeds

Started 2026-10-29 at the user's request, while preparing the final report: add training seeds
to DPO_V3_4880, the run whose result is the cleanest and cheapest so far ("补 43、44，共 3 个").
Branch `xinyang/report-dpo-cost` (from main d2c4fe0). The comparison is
[`EVAL_STANDARD.md`](EVAL_STANDARD.md), unchanged; the reference arm is SFT-v3.

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
| 2 | seed 43: evaluation | 5199636 (commit 1912a3d) | 0.50 [1.21] | done 2026-10-29, 0:18:09, verdict below |
| 3 | seed 44: training | 5199738 | | submitted 2026-10-29 |
| 4 | seed 44: evaluation | | | |

Spent so far: 5.27 GPU-h allocated of the 12 GPU-h cap.

## Result so far (seed 43; seed 44 pending)

Against SFT-v3 at matched hire rate (`reports/reference_metrics.csv`,
`reports/operating_point/v2/Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_seed43_step305--vs--Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional.json`;
served model `outputs/merged/qwen3.5-9b_en_only_dpo_v3_4880_seed43--checkpoint-305`):

| Seed | Hire rate | Δ unstable, own point (pp) | Δ at SFT-v3's 22.51% (pp) | TPR vs GPT-4o vs SFT-v3 (pp) | Merge kept | Guardrails | Verdict |
|---|---|---|---|---|---|---|---|
| 42 | 22.79% | −1.03 [−1.81, −0.28] | −0.81 [−1.56, −0.15] | +0.70 | 95.85% | 4/4 pass | better |
| 43 | 21.59% | −0.09 [−0.88, +0.66] | −0.30 [−1.00, +0.41] | −2.11 | 95.57% | 4/4 pass | not better: the interval reaches 0 |

The common-point column comes from `reports/final/final_table.csv` (`delta_common_pp`,
`ci_common_*`). Seed 43 keeps the sign the For rule asks for, Δ < 0, and passes every
guardrail, so the round is not decided yet; seed 44 decides it.
