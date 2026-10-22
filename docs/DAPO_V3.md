# DAPO from SFT-v3

DAPO-v3-all is GRPO-v3-all (`docs/GRPO_V3.md`, round 3) with DAPO's objective (Yu et al. 2025, arXiv 2503.14476) and a strict balance of the two reference verdicts, on the same prompts. The trainer is `mitigation/dapo.py`, a subclass of TRL's GRPOTrainer that reuses `mitigation/grpo.py`'s rewards, policy, weight sync and monitors. The branch is `xinyang/dapo`, cut from `xinyang/v3-additional-data` at aa8aa57. This page holds the setup, what DAPO asks of the training data, and a pre-registration draft. Nothing has been trained yet.

## What changes from GRPO-v3-all

| | GRPO-v3-all (job 5116822) | DAPO-v3-all |
|---|---|---|
| Token-level loss | Already on: `loss_type: dapo` is TRL 1.15's default (its `training_args.bin`) | Same, set explicitly |
| Clip-higher (ε 0.2 / 0.28) | ε 0.2 both ways, and inert: with one optimizer step per rollout the ratio is 1. `clip_ratio` was 0.000 in every stretch of both GRPO-v3 runs (`scripts/dapo_signal.py log`) | ε_high 0.28, with rollouts of 32 prompts that train two optimizer steps, so the second step is off-policy and the clip can bind. The paper takes 16 steps per rollout |
| Dynamic sampling | None. A group whose decisions all agree has zero advantage and still counts in the token denominator | Only groups with 0 < correct < 16 train, judged by the parsed decision. Up to 8 probe rounds per rollout |
| Verdict balance (not DAPO) | Prompts drawn half per verdict, and every group trains | Each verdict trains as many groups as the scarcer one kept, over all ranks |
| Overlong shaping | None | Soft punishment from 80 of 96 tokens. GRPO-v3-all's longest untruncated answer was 67 tokens, and one of its ~146k completions hit 96, so this is a guard |
| KL term | β 0.04 toward SFT-v3 | β 0 (DAPO drops it) |
| Checkpoints | Every 25 steps | Every 24, a whole number of rollouts: a checkpoint inside a rollout resumes onto a fresh rollout of its prompts |

Unchanged: the prompts (`configs/data/en_only_grpo_v3_all.yaml`), SFT-v3 as the start, LoRA r 32, lr 3e-5, 16 generations, 16 prompts per optimizer step, one epoch of the main sampler, and the monitors. `tests/test_dapo.py::test_dapo_v3_all_differs_from_grpo_v3_all_only_in_the_declared_settings` pins this list.

**How dynamic sampling is built** (`mitigation/dapo.py`, module docstring):

- **Probing.** Probes use TRL's own `_generate` on each rank's colocated vLLM, with the policy's weights at the start of the rollout. Each rank keeps up to 4 informative groups per verdict. While any rank is short, every rank probes a refill batch, at most 8 rounds per rollout.
- **Refills.** They come from a seeded permutation of the training prompts that the ranks share, drawn beside the main sampler. So within one epoch of the main sampler a prompt can train more than once, and `dynamic_sampling.jsonl` lists the prompts each rollout trained.
- **Balance.** Each verdict trains as many groups as the scarcer verdict kept over all ranks.
- **Scoring.** Only the kept rollouts are replayed into TRL's scoring. Groups that make up a short quota, or that the balance cuts, keep their place with a zero loss mask.
- **Stop.** `dapo/fill` is the share of the quota that trains. When its mean over the last 25 rollouts falls under 0.5, the run saves and stops, and a stopped run refuses `resume: true`.
- **Metrics.** TRL's reward metrics in this run describe the trained batch. The readings comparable with GRPO's logs are `dapo/probe_*`, taken on each rollout's first probe, which is the main sampler's draw.

## What DAPO asks of the training data, against GRPO

GRPO takes any prompt with a verifiable target. A prompt on which the policy always agrees costs a rollout and teaches nothing, but it does no harm. DAPO trains only on prompts that split the *current* policy, so the data has to keep supplying them.

1. **A supply of prompts the policy is unsure about.** It matters what the policy being trained is unsure about, not the SFT parent.
   - Under SFT-v3, 87.3% of grpo-v3-all's prompts are expected to split a group of 16 (`scripts/dapo_signal.py prompts`).
   - Under the policy as GRPO-v3-all trained, zero-variance groups rose from 50.4% of the batch in the first eighth to 96.2% in the last.
   - So on the same prompt distribution, the informative share fell from about one half to about 4%.

   | GRPO-v3-all, steps | zero-variance groups | decision accuracy | hire rate, ref hire | hire rate, ref reject |
   |---|---|---|---|---|
   | 1–71 | 0.504 | 0.670 | 0.693 | 0.367 |
   | 143–213 | 0.892 | 0.820 | 0.820 | 0.172 |
   | 285–355 | 0.954 | 0.886 | 0.883 | 0.106 |
   | 498–571 | 0.962 | 0.877 | 0.870 | 0.104 |

   (`scripts/dapo_signal.py log $XW/logs/xw-grpo-5116822.out --parts 8`, every second stretch. GRPO-v3 on its 3,000 prompts went from 0.731 to 0.981.)

2. **A pool much larger than the steps consume.**
   - Dynamic sampling trains 16 informative groups per optimizer step. At a 4% informative share that means about 400 prompts probed per step, and grpo-v3-all's 9,140 prompts would last about 23 such steps. GRPO consumed 16 prompts per step whatever their signal.
   - The paper's verl recipe probes up to 10 batches of 3× the train batch. Here the cap is 8 rounds, at most 128 prompts per optimizer step, so filling half the quota needs an informative share of about 6% on each verdict.
   - A pool that is mostly certain prompts, such as round 3's whole training pool, needs its refills weighted toward rows with a sampled error. Under SFT-v3, 4,570 of its 69,794 reject training rows have one (`artifacts/semisynthetic-v3/grpo-v3-all/report.json`, `available_informative`).
3. **Both verdicts informative.** Without KL, the verdict mix of what trains is what holds the hire rate.
   - The balance cuts training to the scarcer verdict, so that verdict's supply sets both the cost and the fill.
   - Late in GRPO-v3-all the policy still hired 10–21% of reject-anchored samples, against 87–91% of hire-anchored ones.
   - GRPO-v3-all already failed the TPR guardrail by moving toward reject (−10.11 pp).
4. **Reliable targets where the policy is unsure.**
   - The kept groups concentrate on rows where the policy splits. Those are the rows where an 8-draw vote anchor is most likely to be borderline, and each gets a full-strength update.
   - GRPO spreads the same rows over a batch that is mostly zero-advantage.
   - In grpo-v3-all, SFT-v3's mean sampled error rate is 0.354 on hire rows and 0.451 on reject rows. These are hard rows, and many may sit near the anchor's own margin.
5. **Unchanged.** Rewards stay rule-based on a binary decision. DAPO's integer-answer transformation, which removed parser noise in maths, has no counterpart here. Overlong shaping needs nothing, since answers are well under the length cap.

What this suggests for the data, not yet built or tested: a pool several times larger than grpo-v3-all, drawn from the round-1 and round-3 on-policy pools by sampled error rate on both verdicts, and refills weighted the same way. Optionally, rows could be re-scored under a mid-run checkpoint, since SFT-v3's error rates go stale as training proceeds.

## Pre-registration (2026-10-22, before training; settings decided by the user)

- **Question.** Does DAPO with a strict verdict balance, from SFT-v3 on grpo-v3-all's prompts, lower the share of unstable within-group sets at matched hire rate against SFT-v3?
- **Control.** SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`, through `evaluate_arm.sbatch`'s op-margin-v2 compare on all 31,050 prompts and 2,700 sets. GRPO-v3-all (−0.56 pp [−1.61, +0.41], TPR guardrail failed) is context: same prompts, GRPO's objective.
- **For the hypothesis.** Δ < 0 with the 95% interval below 0, and all four guardrails passing.
- **Against it.** An interval that reaches 0 or a Δ above 0, or a guardrail failure.
- **Expected.** If the policy loses its uncertainty as GRPO-v3-all's did, the fill drops and the stop fires before the epoch ends. That is a data result, not a failure of the code.
  - Diagnostics, read from `dynamic_sampling.jsonl` and `dapo/*`: the fill and the probe rounds per rollout, the informative share per verdict, and how often a prompt trains.
- **Stop conditions.**
  - The decision monitor's band, 15–35% projected hire rate.
  - `memory_floor_gb` 12.
  - The mean fill over 25 rollouts (50 steps) under 0.5.
  - An 11 h 15 min limit (45 GPU-h on four GPUs), as GRPO-v3-all; the user is asked before `resume: true` continues it.
- **Selection and test.** The final adapter is audited, whichever stop ends the run. The benchmark audit is spent once.
- **Order.** After GRPO-v3-all's comparison (exploratory step-525 audit, job 5119034) has finished, and with the account's queue otherwise empty: one job at a time. The code is committed before submission.
- **Decided by the user (2026-10-22): "β=0 加严格平衡".**
  - β 0, as DAPO. In GRPO-v3-all, β 0.04 held the mean per-token KL to SFT-v3 at 0.016–0.034 through training (job 5116822's log), and the hire rate where GPT-4o hires still fell 10.11 pp. The KL is averaged over ~38 completion tokens, of which the decision is one or two. Our reading, not measured on the decision token: it held the wording more than the decision.
  - The strict verdict balance stays on. It is not part of DAPO, so a win of this run does not say whether DAPO or the balance did it. The ablation for that is the same run with `dynamic_sampling_balance: false`, a second full training.
- **Decided by the user (2026-10-22): "Use grpo-v3-all's prompts as they are for DAPO-v3-all, and keep max rounds at 8".**
  - The prompts are grpo-v3-all's 9,140, unchanged, so the comparison with GRPO-v3-all is on the same data.
  - `dynamic_sampling_max_rounds` stays at 8.
- **Costs.** Under `dapo_v3_all`.

## Reproducing

From a Daint login node, one submission at a time:

```bash
export REPO=$PWD PYTHONPATH=$PWD/src
python scripts/dapo_signal.py log $XW/logs/xw-grpo-5116822.out --parts 8
python scripts/dapo_signal.py prompts $XW/artifacts/semisynthetic-v3/grpo-v3-all --generations 8 16 32
HBM_COST_ARM=dapo_v3_all CONFIG=configs/mitigation/dapo/qwen3.5-9b_en_only_dapo_v3_all_gh200.yaml \
  scripts/cscs/submit.sh -p normal -t 11:15:00 scripts/cscs/train_dapo.sbatch
HBM_COST_ARM=dapo_v3_all AUDIT=configs/audit/qwen3.5-9b_en_dapo_v3_all_gh200.yaml \
AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh scripts/cscs/evaluate_arm.sbatch
```
