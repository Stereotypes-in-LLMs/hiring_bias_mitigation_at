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

**A data fact behind explanation 3.** On 3,366 of the 9,140 training rows (37%) the vote anchor
and SFT-v3's own attribute-free majority (8 samples at temperature 1, from the round-3 pools)
disagree: 1,580 hire-anchored rows where SFT-v3 rejects without the attribute, and 1,786
reject-anchored rows where it hires (`artifacts/semisynthetic-v3/grpo-v3-all-free/report.json`,
built by `scripts/build_grpo_free_target.py`). On those rows the reward teaches a different
hiring standard, not invariance to the attribute, and the two kinds push the decision bias in
opposite directions, about equally often.

The paired probe of 2026-10-22 points the same way (`$HBM_OUTPUT_ROOT/outputs/paired_probe/
20261022-dapo-final-sft-v3-k16/report.md`; 16 samples of each training row with and without
the attribute). Rows where all 16 attributed samples miss the anchor rise from 336 under SFT-v3
to 1,170 under DAPO-v3-all's final adapter, and 1,081 of the 1,170 also miss it on all 16
attribute-free samples. What RL moved on its own training rows is mostly the case judgement,
not the response to the attribute.

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

### A2 — consistency reward (method)

- **Hypothesis.** Rewarding agreement with the model's own attribute-free answer targets the
  evaluated quantity (explanation 3), and on the 37% of rows where the anchor contradicts that
  answer it cancels the push toward a new standard, which is also a source of the bias swing.
- **Change.** A third reward, `grpo.free_consistency_reward`: SFT-v3's sampled attribute-free
  hire share for the pair when the completion hires, one minus it when it rejects. Summed with
  the decision and format rewards before the usual group normalisation (`sum_then_normalize`,
  weight 1 each). The data are grpo-v3-all's rows, order and splits unchanged, plus
  `free_hire_rate` (`configs/data/en_only_grpo_v3_all_free.yaml`). Static target: SFT-v3's
  answer, not the current policy's; the plan's on-policy pair blocks
  (`GDPO_CONSISTENCY_PLAN.md`) are left for later. Config
  `configs/mitigation/grpo_improve/a2_consistency.yaml`; tests `tests/test_grpo_free_target.py`.
- **Expected if it holds.** Mean proxy matched change at least 0.3 pp better than A0's, with TPR
  change against SFT-v3 within ±5 pp at every scored checkpoint.
- **Against it.** A proxy matched change within 0.3 pp of A0's or worse: the conflicting rows
  are not what keeps GRPO from adding invariance. A TPR drop past −8 pp: the target pulls the
  operating point.
- **Cost.** About 11 GPU-h; the reward adds a lookup.

### Later candidates, not registered

- A1 + A2 together, if each helps on its own.
- A bias anchor on held-out Djinni prompts in SFT-v3's 10–90% band (penalise the policy's mean
  decision margin moving from SFT-v3's), if A1 shows the drift is not first-order.
- Twice the prompts per step at the same lr (explanation 2 through noise), if A0's swing is
  large and A1 does not remove it.

## 5. Round 1 result (A0, A1, A2)

**In one sentence.** None of the three short runs lowers the proxy matched instability against
SFT-v3, and the bias projection doubled the TPR swing instead of removing it. Development
readings on the exposed benchmark, one seed each.

| arm (200 steps from SFT-v3) | TPR SD, 100–200 | largest neighbour TPR change | TPR change range vs SFT-v3 | checkpoints < −10 pp | mean proxy matched change | proxy range | raw unstable, mean | zero-std groups, mean |
|---|---|---|---|---|---|---|---|---|
| GRPO-v3-all, seed 42 (history, job 5127818) | 6.8 pp | 15.2 | −10.2 to +5.0 | 1 | −0.53 pp | −0.88 to +0.04 | 1.71% | 0.81 |
| A0 control, seed 43 | 5.9 | 13.9 | −2.5 to +11.4 | 0 | −0.12 | −0.55 to +0.39 | 1.80% | 0.79 |
| A1 bias projection | **12.2** | **27.3** | −15.6 to +16.2 | 2 | −0.05 | −0.65 to +0.24 | 1.75% | 0.78 |
| A2 consistency reward | 5.2 | 7.7 | −5.6 to +7.5 | 0 | +0.05 | −0.46 to +0.97 | 2.04% | 0.85 |

Source: `$HBM_OUTPUT_ROOT/outputs/grpo_improve/compare_round1/{summary.csv,curves.csv,curves.svg}`;
per checkpoint `outputs/grpo_improve/diag_{a0,a1,a2}/trajectory_diagnostics.csv` and
`transitions.csv`; raw decisions `outputs/checkpoint_benchmark/grpo_improve_{a0_seed43,
a1_bias_projection,a2_consistency}/steps/step-NNNN/raw.parquet`. SFT-v3: TPR 57.73%, raw
unstable 1.85%.

**Against each registration.**

- **A0.** For: the swing reproduces at seed 43 (SD 5.9 pp ≥ 4), at other steps and in the other
  direction (lenient: +11.4 pp at step 200 against seed 42's −10.2 at 150). Against: the
  per-verdict counts of mixed groups do not predict the sign of a window's TPR change. Over
  176–200 both verdicts had 25 mixed groups and TPR rose 13.9 pp; correlation of the net count
  with the TPR change over the five windows −0.14 (`rollout_diagnostics.jsonl`). In training
  rollouts, the sampled hire rate where the anchor hires fell from 0.58 to 0.01 by step ~19 and
  was back at 0.49 by step ~28 (job 5140665's log).
- **A1. Hypothesis refuted.** With the first-order push on a shared decision bias set to zero
  every step (logged `push_after` ~1e-15), the swing doubled. The registered reading: the
  drift does not come from the first-order bias push. The likelier role of that push is a
  restoring force: when the policy drifts to reject, the hire-anchored groups become
  informative and push back. Removing it removed the brake, not the engine.
- **A2. Not better.** Mean proxy +0.05 against A0's −0.12 (registered: at least 0.3 pp better),
  and TPR leaves ±5 pp at two checkpoints. It changes fewer decisions against SFT-v3 (656–1,302
  against A0's 1,121–2,031) and its largest neighbour change is about half A0's.

**What this says about the three explanations.**

- *Rollout signal (1).* Unchanged by any arm: 0.78–0.85 zero-std groups over 100–200.
- *Update size (2).* The operating point moves by several pp per 25 steps in every arm, and
  not through the decision-bias gradient. Candidates left: Adam's per-parameter scaling with a
  constant lr (the update does not shrink when the signal does, and its direction is not the
  gradient's), the KL term, or learning on feedback tokens. Not separated yet.
- *Target mismatch (3).* Supported again: the training-rollout hire rates and the benchmark TPR
  stay uncorrelated in every arm (r between −0.14 and 0.19).
- *Seed noise in the readout.* The same config at seeds 42 and 43 gives mean proxies of −0.53
  and −0.12. A difference between two single-seed short arms below ~0.5 pp is not readable.

**Two facts from existing outputs, CPU only.**

- Over GRPO-v3-all's whole history the proxy is −0.53 pp on average at 100–200, +0.09 at
  225–400 and −0.83 at 425–571, where all seven checkpoints sit at −0.70 or below while TPR
  swings by 30 pp (`compare_history/`). Whatever gain GRPO has shows late and survives the
  operating-point swing.
- The mean-of-7 weight average (`checkpoint_average/.../average/raw.parquet`): TPR −3.0 pp,
  proxy −0.32 pp [−1.37, +0.67]. Averaging centred the operating point and lost about half of
  the late checkpoints' gain.

## 6. Round 2: registrations

Both arms start from SFT-v3 for 200 steps and are read as in section 3, against A0.

### R2-a — operating-point anchor (method)

- **Hypothesis.** The benchmark's threshold drifts because nothing holds it once the training
  rows are settled (round 1: A1's removal of the restoring push doubled the swing). A restoring
  force on held-out prompts like the benchmark's borderline ones holds it, and the late GRPO
  gain does not depend on the drift.
- **Evidence before running.** On the monitor calibration's seven finished arms (job 5123830),
  the mean SFT-relative margin shift on these 222 rows correlates 0.93 with the arm's
  benchmark TPR change (development evidence, 30 candidates).
- **Change.** A0's config plus `operating_point_anchor` (`mitigation/grpo_operating_anchor.py`):
  each micro-step and rank forwards 4 anchor prompts with the decision prefix, and the loss
  gains 10 · (mean p_policy − mean p_SFT)², p the hire probability of the decision token, p_SFT
  with the LoRA disabled. The anchor rows (`scripts/build_grpo_anchor_rows.py` →
  `artifacts/semisynthetic-v3/grpo-anchor-djinni-band/`) are the 222 Djinni monitor rows, 30
  held-out candidates, with SFT-v3's margin in its 10–90% band; none is in training or the
  benchmark. Weight 10 is set before running: a shift of 0.1 in mean hire probability costs
  0.1, five times GRPO-v3-all's typical loss. Logged as `anchor/shift`, `anchor/penalty`.
- **Expected if it holds.** TPR SD over 100–200 under 3 pp, every checkpoint within ±8 pp, and
  the proxy matched change within 0.3 pp of A0's or better. `anchor/shift` stays under 0.05.
- **Against it.** A swing within 30% of A0's (5.9 pp): held-out Djinni rows do not carry the
  benchmark's threshold. A stable TPR with a proxy worse than A0's by more than 0.5 pp: the
  anchor costs the fairness gain.
- **Check before the run.** A three-step smoke on `debug` (`r2a_anchor_smoke.yaml`) for the
  anchor's forward and backward under DDP. CPU tests: `tests/test_grpo_operating_anchor.py`.
- **Cost.** About 12 GPU-h with the smoke; the anchor adds 4 short forwards per micro-step.
- **Amendment after a failure, before any result (2026-10-25, 15:30).** Run 5141612 hung at step
  2: a DDP bucket all-reduce that rank 3 never reached, NCCL timeout, 2.72 GPU-h lost; nothing
  is evaluated. The penalty was added to each micro-step's loss with its forward through the
  unwrapped module beside DDP's; the three-step smoke (5141458, 0.79 GPU-h) had passed. The
  anchor is now a callback after gradient clipping (`grpo_operating_anchor.OperatingAnchorCallback`):
  its gradient comes from `torch.autograd.grad`, which DDP's hooks never see, and is summed over
  ranks with one all_reduce per step. `rows_per_step` now counts rows per optimizer step and rank:
  8, so 32 rows a step against the first version's 128, at a quarter of its forward cost. The
  weight, rows, hypothesis and criteria are unchanged. A six-step smoke checks it first. This
  is retry 1 of 3.
- **Second amendment, from the smoke, before any result.** The six-step smoke (5142306, 0.83
  GPU-h) ran clean at 25–35 s a step. Its `anchor/shift` changed sign every step at about
  ±0.014, the sampling noise of 32 rows, while the anchor's gradient norm was 3–7 against
  the policy gradient's 0.5–0.8 after clipping. At weight 10 the anchor would add a noise
  direction ten times the policy's every step. The weight is now **1**: about the policy
  gradient's size at noise level, and about 4.6 at a real shift of 0.1.

### R2-b — a third of the learning rate (parameter)

- **Hypothesis.** The operating point moves because each update is too large for the signal
  (explanation 2), whatever its direction.
- **Change.** `learning_rate: 1.0e-05` on A0's config, nothing else (`r2b_lr1e5.yaml`).
- **Expected if it holds.** TPR SD under 2.5 pp and fewer decisions changed against SFT-v3,
  roughly in proportion to the learning rate.
- **Against it.** A swing within 30% of A0's: the step size does not set the swing, and the
  walk comes from something that does not scale with lr (Adam's normalised steps hold
  their size only in proportion to lr, so this would point at the data or the sampling).
- **What it cannot be.** An arm of the standard: the bf16 merge will likely keep under 80% of
  so small an update (the pilot at 1e-5 kept 58%). Its checkpoints are scored with a declared
  floor of 30% (`min_merge_retention` in its trajectory config) and the kept share is reported
  beside each reading. The served model is then only part of the trained one, which weakens
  any null.
- **Cost.** About 11 GPU-h.

## 7. Ledger

| # | Arm | Job(s) | GPU-h | State |
|---|---|---|---|---|
| — | CPU diagnosis (section 2) | none | 0 | done |
| 1 | A0 control: training, scoring | 5140665, 5141030 | 8.47 + 2.03 | done |
| 2 | A1 bias projection: training, scoring | 5140670, 5141105 | 8.44 + 2.04 | done, refuted |
| 3 | A2 consistency reward: training, scoring | 5140696, 5141121 | 8.40 + 2.03 | done, not better |
| | **Total** | | **31.41 of 150** | 3 of 50 experiments, 0 retries |
