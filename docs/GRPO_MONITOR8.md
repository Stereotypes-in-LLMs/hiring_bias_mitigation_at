# GRPO with a calibrated 8 pp TPR stop

## Setup

User authorization, 2026-10-22: first calibrate the 696 held-out monitor rows. Only if calibration passes, train a GRPO comparison on the original prompt and 9,140 rows with an 8 pp projected-TPR stop. Do not retrain the other models.

Question: can the new monitor detect the TPR losses missed by the original monitor, and can its stop rule preserve the original GRPO arm's stability benefit without exceeding the TPR guardrail?

The parent and final comparison reference are SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`. The existing GRPO-v3-all run is the training-procedure control. DAPO and the other completed arms are transfer checks only. This is one seed and one GRPO run, conditional on calibration.

The original binary prompt, rewards, 9,140 training rows, 256 validation rows, SFT-v3 checkpoint, optimizer, rollout settings and data split remain the control settings. The 696 monitor rows come from 222 held-out candidates. No v1 evidence-ledger prompt, review output or teacher-correction candidate is used in this experiment.

## Calibration acceptance, fixed before GPU scoring

The registered config is `configs/calibration/grpo_v3_all_monitor8.json`. It pins 13 source files, including the training/validation/monitor Parquets, parent benchmark responses, all eight benchmark margin files and the existing generated-response metrics. Eight existing merged models score the same ordered monitor prompts with the existing op-margin-v2 decision scorer.

Every condition below must pass:

1. Exactly 696 finite scores per model, identical ordered prompt digest and model identity, with 222 monitor candidates and no training-candidate overlap. The original training set has 9,140 rows and the registered hashes still match.
2. Every benchmark-weighted bin has at least one row when binned by the freshly scored SFT-v3 decision margins. Report both rows and distinct candidates per bin. Sampled hire-rate bins in `monitor.json` do not establish this coverage.
3. For all seven other arms, absolute error between monitor-projected TPR change and actual attributed-benchmark margin TPR change is at most 2.0 percentage points.
4. For every arm, optimism against its existing all-condition generated-response TPR change is strictly less than 2.0 pp. Optimism is projected change minus actual change, so a positive value hides some of the loss.
5. Any arm whose observed margin-based or generated-response TPR change is below -8 pp must also have a projected change below -8 pp. This includes DAPO final, GRPO final and GRPO checkpoint 525.

FPR errors and per-bin row and candidate counts are reported. They do not replace these predeclared TPR gates. Weights come only from the SFT-v3 benchmark margin distribution, using the existing calibration formula; no arm's transfer result is used to refit weights or move thresholds. A failed gate means no training submission.

This checks transfer on completed arms, not every future optimization trajectory. It also accounts for the difference between attributed margin TPR and the final guardrail's all-condition generated-response TPR. It is not a guarantee that a run cannot cross the guardrail between the 25-step checks.

## Conditional training and comparison

If calibration passes, create a new output directory and GRPO config cloned from `qwen3.5-9b_en_only_grpo_v3_all_gh200.yaml`. Add the frozen SFT-only `margin_monitor` calibration, `steps: 25` and `tpr_drop_stop: 0.08`; start fresh rather than resuming the old run. Keep the original 15–35% hire-rate band and memory floor as additional stop conditions.

Budget: one epoch, approximately 571 optimizer steps, with an 11 h 15 min wall-time cap on one four-GH200 node, at most 45 allocated GPU-hours. The calibration has a separate 45-minute cap on the same node size. No automatic extension or second seed is included.

Checkpoint selection is declared before training: use the latest saved checkpoint that has a recorded margin-monitor reading at or above -8 pp and before any safety stop. Keep the checkpoint that triggers a stop for diagnosis. If the first monitored checkpoint already breaches the line, report that no eligible new GRPO checkpoint exists and retain SFT-v3 as the baseline. Do not select using benchmark results.

Evaluate the selected checkpoint once using the original binary audit protocol and `docs/EVAL_STANDARD.md`: matched-hire-rate instability versus SFT-v3, TPR/FPR, agreement and undecided coverage. Report actual training steps and the stopping reason alongside the existing GRPO-v3-all result. An interval touching zero or any failed guardrail does not support improvement.

## Result

Pending. This document and calibration config are recorded before monitor scoring. Training is conditional on the acceptance gate, not on a successful scoring process alone.

## Reproducing

The calibration job uses `scripts/cscs/monitor_calibration.sbatch` with the eight registered `LABEL=model` entries, a unique `SCORES_DIR`, and the seven registered `LABEL=benchmark-run` comparisons. `scripts/check_monitor_transfer.py` will verify source integrity and write the machine-readable acceptance result. A nonzero gate exit prevents training.

All new outputs are under `outputs/monitor_calibration/grpo_v3_all_monitor8_20261022`. Original models and original train/validation/monitor data are retained. Code commit, job ID, logs, score files and gate result will be recorded here when available.
