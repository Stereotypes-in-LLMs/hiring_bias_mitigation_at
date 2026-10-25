# GRPO-improve: short experiments on why GRPO from SFT-v3 is unstable

Started 2026-10-25 on `xinyang/grpo-improve` (from main 4f9f970, with `xinyang/dapo` merged to
4b5061a for its checkpoint-trajectory scorer and rollout diagnostics). The user's goal: lower the
share of unstable decisions while keeping TPR and utility, under the comparison of
[`EVAL_STANDARD.md`](EVAL_STANDARD.md) unchanged. Training reward is a diagnostic, never the
success criterion.

**Budget (user, 2026-10-25).** 150 GPU-h in total, sampling, failed attempts and evaluation
included; at most 50 short experiments; at most 3 jobs at once and 3 retries. Costs are recorded
under `HBM_COST_ARM=grpo_improve`. The ledger is at the end of this page.

## 1. What is already known

| Work | State | Result | Source |
|---|---|---|---|
| GRPO-v3 (SFT-v3's own pairs, 187 steps) | done, valid | not better: +0.74 pp [−0.15, +1.63]; guardrails pass | `GRPO_V3.md` |
| GRPO-v3-all final, step 571 | done, valid | not better: −0.56 pp [−1.61, +0.41]; **TPR guardrail fails**, −10.11 pp | `GRPO_V3.md` |
| GRPO-v3-all checkpoint 525 | done, exploratory (chosen after the audit) | better: −1.02 pp [−1.99, −0.12]; TPR −9.30 pp; the user recorded GRPO as better | `GRPO_V3.md` |
| DAPO-v3-all (β 0, strict verdict balance) | done, no improvement | +0.85 pp [−0.23, +2.12]; TPR fails, −10.10 pp; pool ran dry at step 85 | `DAPO_V3.md` |
| Checkpoint trajectory, 20 checkpoints of GRPO-v3-all | done, valid diagnostic | full-benchmark TPR swings between −14.4 and +15.6 pp of SFT-v3's from one saved checkpoint to the next; first −10 pp failure at step 150 | `GRPO_CHECKPOINT_TRAJECTORY.md`, job 5127818 |
| Mean of the 7 late checkpoints' updates | done, no improvement on its preregistered test | c central, d′ below step 500's (−0.31 [−0.53, −0.10]); no margin pass, so no fairness verdict | `GRPO_CHECKPOINT_WEIGHT_AVERAGE_RESULTS.md`, job 5130000 |
| Margin monitor, 696 held-out rows | done, **calibration failed** | projects 5–6 pp too little TPR loss for DAPO, GRPO-v3-all and step 525; the RJDB/Djinni source mix explains most of it | `GRPO_MONITOR8.md`, job 5123830 |
| Direct benchmark-dev monitor (90 candidates) | done, **failed its replay** | dev −8 pp first fires at step 400; the full benchmark already failed −10 pp at 150 | `BENCHMARK_DEV_MONITOR.md` |
| Lower LR 3e-6 from checkpoint 425 | **failed twice before any update** | 5130709: a W&B startup race between ranks. 5131944: the startup gate found 52 of 31,050 decisions differing from the frozen cache; zero updates | `GRPO_LOW_LR_PILOT*_RESULTS.md` |
| GRPO-v3-new | stopped by the user at step 4 | nothing | `GRPO_V3.md`, round 2 |

Not yet tested at all: the effect of a lower learning rate, a reward that targets consistency
across variants (`GDPO_CONSISTENCY_PLAN.md` is a plan), a larger batch, a second seed of
GRPO-v3-all, and DAPO's verdict balance apart from its β 0.

## 2. Diagnosis from what is on disk (no GPU)

`scripts/grpo_trajectory_diagnostics.py` reads the 20 checkpoints' benchmark decisions (job
5127818), SFT-v3's op-margin-v2 margins and GRPO-v3-all's `log_history`. Outputs are under
`$HBM_OUTPUT_ROOT/outputs/grpo_improve/diag_grpo_v3_all/` (`trajectory_diagnostics.csv`,
`transitions.csv`, `train_log.csv`, `summary.json`).

It adds a **greedy proxy of the decision number**: the checkpoint's greedy decisions stand in for
its margin pass, and SFT-v3 is moved to the same hire count by
`operating_point_control.matched_comparison`, unchanged. Checked against the two official
reports: step 525, proxy −1.04 against official −1.02 pp; step 571, −0.71 against −0.56 pp;
SFT-v3 against itself, −0.07. It is a development reading, not a verdict.

**Explanation 1, rollouts without reward variance: supported.** The share of groups whose 16
rewards are identical is 0.58 over steps 1–100 and 0.83 over 101–125, and stays at 0.86–0.98
from step 150 on. A step trains 16 groups, so from about step 150 on a step has two or fewer
groups with any advantage.

**Explanation 2, updates too large for the signal: supported, in a specific form.** The
benchmark decisions move as one global threshold:

- In 14 of the 20 windows between saved checkpoints, at least 95% of the changed decisions go
  the same way (`one_way_share`). Steps 475→500 change 3,111 decisions, all to hire; steps
  500→525 change 3,232, all to reject.
- Criterion c ranges 0.41–0.99 across checkpoints, while d′ stays within 1.64–2.06.
- The proxy matched change wanders within −1.07 to +0.72 pp without a trend. At step 100 it is
  −0.64, at 571 −0.71.
- The changed decisions concentrate on prompts SFT-v3 is unsure about: 41–95% of them lie
  in SFT-v3's 10–90% hire-probability band, which holds 8.1% of all prompts.

**Explanation 3, the training distribution is not the evaluation target: supported.** On the
training rollouts the sampled hire rate where the anchor hires rises from 0.72 to 0.82–0.97 and
stays there, while the benchmark TPR swings. Across checkpoints the two are uncorrelated
(r = −0.04; window change against window change, r = −0.22). Training reward climbs, the
matched instability does not move.

**One reading that ties them together, not confirmed.** By step ~125 the policy is sure of
almost every training prompt. From then on neither the reward nor the KL term (β 0.04 on a
per-token mean of ~38 tokens) holds the hire/reject bias where the benchmark's borderline
prompts sit, because no training prompt is near its own boundary. One or two informative groups
a step, at a constant lr of 3e-5, then move that bias at random. The operating point an audit
sees depends on where the walk happened to stop. That explains the TPR guardrail failures and
the verdict flip between checkpoints 525 and 571.

**A constraint on the obvious fix.** A lower learning rate meets the merge guardrail: the GRPO
pilot at 1e-5 kept 58% of its update in the bf16 merge, below the standard's 80%. A short run at
lower LR from SFT-v3 would not pass the merge check that the scorer enforces. That is why the
low-LR pilot continued from checkpoint 425.

## 3. The short-run protocol, shared by every arm

- **Start.** SFT-v3 (`outputs/sft/qwen3.5-9b_en_only_sft_v3_additional`), fresh LoRA, GRPO-v3-all's
  data (`configs/data/en_only_grpo_v3_all.yaml`, 9,140 prompts) and settings, unless the arm says
  otherwise. 200 optimizer steps; checkpoints every 25.
- **Readout.** Checkpoints 100, 125, 150, 175 and 200, scored on all 31,050 benchmark prompts by
  `scripts/checkpoint_benchmark_trajectory.py` (merged bf16, the audit's decoding, the 80% merge
  gate). For each: full TPR change against SFT-v3, the greedy proxy of the matched change, c, d′,
  raw unstable sets; dev and historical remainder separately.
- **Summary per arm, fixed now.**
  - TPR swing: the standard deviation of full TPR over the five checkpoints, and the largest
    change between neighbours. GRPO-v3-all at 100–200: 6.8 pp and 15.2 pp.
  - Mean proxy matched change over the five. GRPO-v3-all at 100–200: −0.52 pp.
  - Checkpoints below −10 pp TPR. GRPO-v3-all at 100–200: one (150).
- **What it is.** The benchmark has been read many times during development, so these are
  development readings on an exposed set, not tests. An arm that looks better goes to the
  standard's `evaluate_arm.sbatch` at a checkpoint declared before that audit, and is labelled
  exploratory there too.
- **Cost.** Training about 2.1 h on one 4-GH200 node (37 s a step) plus start-up, about 9 GPU-h;
  scoring five checkpoints about 2 GPU-h. About 11 GPU-h an arm.

## 4. Arms

Each arm is registered before it is submitted. Parameter arms change one setting of the control;
method arms are listed apart.

### A0 — control: GRPO-v3-all, seed 43 (parameter)

- **Hypothesis.** The TPR swing is a random walk driven by rollout noise, not by the data order.
- **Change.** `seed: 43` (data order, rollouts and monitor rows), `max_steps: 200`,
  `resume: false`, `rollout_diagnostics: {enabled: true}` (per-verdict counts; logging only).
  Config `configs/mitigation/grpo_improve/a0_seed43.yaml`.
- **Expected if it holds.** A TPR swing of the same size as seed 42's (SD ≥ 4 pp over 100–200),
  at different steps. The per-verdict counts of groups with an advantage, summed over a window,
  predict the sign of the TPR change over that window.
- **Against it.** A swing under 2 pp SD (the oscillation was seed 42's), or the same pattern at
  the same steps (data order drives it).
- **Cost.** About 11 GPU-h.
- **Why first.** Every other arm is read against it, and seed 42's checkpoints 25–75 were not kept.

### A1 — bias projection (method)

- **Hypothesis.** The TPR swing is the first-order push of each step on one shared decision bias:
  with one or two informative groups a step, the step's net push toward hire or reject is close
  to a coin flip. Removing that push removes the swing, and leaves the learning that orders
  prompts against each other.
- **Change.** `bias_projection: {enabled: true}` on A0's config (`mitigation/grpo_bias_projection.py`).
  With s_i = ±1 for a hire or reject decision, the gradient of the policy loss with respect to a
  bias on every prompt's decision logit is ½ Σ A_i s_i. Each step subtracts α (s_i − mean_g s)
  from every advantage, with one α for the whole gathered batch, so that Σ A′_i s_i = 0. Group
  sums stay zero, groups whose decisions agree are untouched, and a batch with informative
  groups on one verdict only still trains the difference between them. Detailed rollout
  diagnostics are off (the plain trainer only); `bias_projection/*` is logged instead. Config
  `configs/mitigation/grpo_improve/a1_bias_projection.yaml`; CPU tests
  `tests/test_grpo_bias_projection.py`.
- **Expected if it holds.** TPR SD over checkpoints 100–200 under 2 pp and no checkpoint below
  −8 pp, against A0's swing. The mean proxy matched change no worse than A0's (within 0.3 pp).
  The logged `push_before` changes sign from step to step without a trend.
- **Against it.** A swing within 30% of A0's: the drift does not come from the first-order bias
  push, and some other direction (feedback wording, the KL term, second-order effects) moves the
  operating point. Or a stable TPR with a proxy matched change clearly worse than A0's (by more
  than 0.5 pp): the bias push carried the fairness gain.
- **Cost.** About 11 GPU-h; the projection itself is two gathers a step.

### A2 — consistency reward (method; registered before submission)

Planned: `GDPO_CONSISTENCY_PLAN.md`'s reward, agreement with the model's own attribute-free
answer within a pair block, with `normalize_then_sum`. It targets the evaluated quantity
(explanation 3) and gives variance on prompts the anchor reward no longer splits (explanation 1).

## 5. Ledger

| # | Arm | Job(s) | GPU-h | State |
|---|---|---|---|---|
| — | CPU diagnosis (section 2) | none | 0 | done |
| 1 | A0 control, training | 5140665 | | running |
