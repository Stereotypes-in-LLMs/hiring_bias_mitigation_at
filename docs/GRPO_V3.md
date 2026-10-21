# GRPO from SFT-v3

GRPO-v3 runs the GRPO pilot's recipe (`docs/GRPO_PILOT.md`) from SFT-v3, the 9B English reference arm, as the RL arm beside DPO-v3 (`docs/V3_NEW_DATA.md`). The trainer is `mitigation/grpo.py`, the prompt set `scripts/build_grpo_prompts.py` on V3's pool, and the branch `xinyang/v3-additional-data`. This page is the pre-registration, written before training. The result follows below it.

## Pre-registration (2026-10-21, before training)

- **Question.** Does GRPO from SFT-v3 lower the share of unstable within-group counterfactual sets at matched hire rate against SFT-v3?
- **Hypothesis.** Yes. From SFT v1 the pilot reached −0.95 pp [−1.90, −0.05]. DPO-v3 from SFT-v3 did not (+0.94 pp [−0.26, +2.24], job 5114015).
- **Control.** SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`, the reference arm for anything trained after SFT (`docs/EVAL_STANDARD.md`). The comparison is `evaluate_arm.sbatch`'s op-margin-v2 compare-arms on all 31,050 prompts and 2,700 sets. DPO-v3 and the GRPO pilot are context, not the decision.
- **For the hypothesis.** Δ < 0 with the 95% interval below 0, and all four guardrails passing.
- **Against it.** An interval that reaches 0 or a Δ above 0. Or a guardrail failure. The likely one is the hire rate where GPT-4o hires: the pilot's fell 15.2 pp against its parent, and the limit is −10 pp.
- **Stop conditions.** The decision monitor stops the run when the projected benchmark hire rate leaves 15–35% (SFT-v3 is at 22.75%). `memory_floor_gb` 12 applies. The time limit is 3 h; `resume: true` lets one more submission finish the epoch.
- **Selection and test.** The audited checkpoint is the final adapter, step 187, declared now. The 256 held-out monitor rows and the sync check read only held-out pool candidates. The benchmark audit is spent once.

## Setup

| | |
|---|---|
| Policy | Qwen3.5-9B with SFT-v3's adapter merged in (`outputs/sft/qwen3.5-9b_en_only_sft_v3_additional`, job 5107673), plus a fresh LoRA (r 32, α 64, no dropout) |
| Reference for the KL term | SFT-v3 (the LoRA disabled); β 0.04 |
| Prompts | `grpo-v3` (`configs/data/en_only_grpo_v3.yaml`): 3,000 attributed prompts from V3's pool, 1,500 per vote-anchor verdict |
| Reward | as the pilot: +1.0 for the anchor's decision, +0.2 for the exact JSON with a 4-to-45-word feedback |
| Rollouts | as the pilot: 8 per prompt, temperature 1, at most 96 tokens, one colocated vLLM engine per GPU |
| Optimisation | lr **3e-5** constant after 5 warm-up steps, 16 prompts per step, one epoch: 187 steps |
| Config | `configs/mitigation/grpo/qwen3.5-9b_en_only_grpo_v3_gh200.yaml`; audit `configs/audit/qwen3.5-9b_en_grpo_v3_gh200.yaml` |

These settings differ from the pilot:

- **The learning rate is 3e-5, not 1e-5.** The pilot's bf16 merge kept 58% of its update (mean |ΔW| 8.7e-6), below the standard's 80% merge guardrail. r2em-100 kept 91% at 2.4e-5. A larger update is also the pilot's own third recommendation.
- **The monitor projects onto SFT-v3's benchmark rates.** These are 57.73% where GPT-4o hires and 5.266% where it rejects (`reports/reference_metrics.csv`), through the new `sft_benchmark` config key. The pilot used SFT v1's 65.8% and 9.1%.
- **The band is 15–35%,** around SFT-v3's 22.75%, instead of 20–40% around SFT v1's 28.0%.

**The prompt set** (`artifacts/semisynthetic-v3/grpo-v3/report.json`) uses grpo-02's construction on `semisynthetic-v3/onpolicy-full` and the k=32 hire top-up `onpolicy-hire32`. The targets are the pool's vote anchors, and each row's p is SFT-v3's own sampled error rate. Both halves carry the same summed sqrt(p(1−p)): 368.9, against the pilot's 390.7. All 1,500 hire rows are tipped, and 1,126 of the reject rows. The error mass is 117.5 on the hire half and 191.6 on the reject half. 735 candidates of the 11 `*validation.parquet` files under semisynthetic-v3 are held out (5,256 rows). The 256 monitor rows, 128 per verdict, come from them. Prompts are at most 2,426 tokens (p99 2,066).

**What is new for GRPO here.** SFT-v3 matches its anchors on 98.8% of its samples (job 5108751), against SFT v1's ~85%. Only rows with a tipped sample give a group any variance, and 1,964 hire and 1,440 reject rows are tipped in the training pool. The prompt set uses most of them.

## Reproducing

From a Daint login node, one submission at a time:

```bash
export REPO=$PWD PYTHONPATH=$PWD/src
python scripts/build_grpo_prompts.py --pool artifacts/semisynthetic-v3/onpolicy-full \
  --hire-topup artifacts/semisynthetic-v3/onpolicy-hire32 \
  --validation-root artifacts/semisynthetic-v3 --out artifacts/semisynthetic-v3/grpo-v3
HBM_COST_ARM=grpo_v3 CONFIG=configs/mitigation/grpo/qwen3.5-9b_en_only_grpo_v3_gh200.yaml \
  scripts/cscs/submit.sh -p normal -t 03:00:00 scripts/cscs/train_grpo.sbatch
HBM_COST_ARM=grpo_v3 AUDIT=configs/audit/qwen3.5-9b_en_grpo_v3_gh200.yaml \
AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh scripts/cscs/evaluate_arm.sbatch
```
