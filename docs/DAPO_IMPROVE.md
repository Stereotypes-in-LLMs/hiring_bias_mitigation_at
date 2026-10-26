# DAPO-improve: DAPO-v3-all with GRPO-improve's learning-rate shape

Started 2026-10-26 on `xinyang/dapo` (with `xinyang/grpo-improve` merged at b39c501), at the
user's request: "用这个lr配置，剩下的时间探索DAPO让它更好". "This lr configuration" is GRPO-improve
R3's (`GRPO_IMPROVE.md`): the learning rate held at 3e-5, then decayed linearly to 0 over the
last 26% of the run. R3 was recorded as better than SFT-v3 on the user's decision (−0.71 pp
[−1.89, +0.39], every guardrail passing). The comparison is [`EVAL_STANDARD.md`](EVAL_STANDARD.md),
unchanged; the reference arm is SFT-v3.

**Budget.** What is left of GRPO-improve's 150 GPU-h, 3 jobs at once, 3 retries (one used):
63.0 GPU-h at the start of this page. Costs go under `HBM_COST_ARM=dapo_improve`.

## 1. What DAPO-v3-all did (inference only, job 5147613)

DAPO-v3-all (`DAPO_V3.md`; training 5119438, evaluation 5120574) stopped dry at step 85 and
is not better: +0.85 pp [−0.23, +2.12], TPR −10.10 pp (guardrail fails). Its checkpoints 24, 48
and 72 were scored on all 31,050 benchmark prompts by `scripts/checkpoint_benchmark_trajectory.py`
(`configs/analysis/dapo_v3_all_trajectory.json`, declared merge floor 30%: the merges kept
62.5%, 77.8% and 85.0% of the updates); step 85 is the audit's own output.
`scripts/grpo_trajectory_diagnostics.py` → `$HBM_OUTPUT_ROOT/outputs/dapo_improve/diag_dapo_v3_all/`.

| step | TPR vs SFT-v3 | proxy matched change | raw unstable | trained-batch grad norm (pre-clip, window mean) |
|---|---|---|---|---|
| 24 | +1.0 pp | −0.58 pp | 1.74% | 0.59 |
| 48 | −8.7 | −0.36 | 1.67% | 1.00 |
| 72 | −8.3 | −0.40 | 1.56% | 2.94 |
| 85 | −10.1 | +0.93 (official +0.85) | 2.22% | 7.46 |

- The TPR loss is one move, 24→48: 1,442 decisions change, 98% of them to reject. It then
  stays at −8 to −10 pp: a level shift, not GRPO's back-and-forth swing.
- The instability is lost in the last 13 steps: the proxy goes from −0.40 to +0.93 while
  the gradient norm before clipping climbs from ~1 to 7.5, as the dry pool leaves fewer groups
  per step (DAPO normalises by the kept tokens) and β = 0 leaves nothing to hold the policy.

## 2. D1 — R3's learning-rate shape on DAPO's own horizon (parameter)

- **Hypothesis.** Decaying the learning rate over DAPO's last 26% removes the late damage (the
  72→85 loss of stability) and keeps checkpoint 72's state, at −8 pp TPR, inside the guardrail.
- **Change.** DAPO-v3-all's config with `max_steps: 84` (it stopped dry at 84–85) and
  `warmup_stable_decay`, `num_decay_steps: 22`: 3e-5 to step 62, then linear to 0 at 84.
  Nothing else (`configs/mitigation/dapo/dapo_improve_d1_decay.yaml`). GRPO training was
  bit-deterministic at a fixed seed (`GRPO_IMPROVE.md`, R3); if DAPO's is too, steps 1–62
  repeat DAPO-v3-all's exactly, and the checkpoints 24 and 48 must hash the same. That is
  checked, not assumed.
- **Control and decision.** SFT-v3 by `evaluate_arm.sbatch`; the final adapter (step 84, or the
  dry stop if it comes first), declared now; audit
  `configs/audit/qwen3.5-9b_en_dapo_improve_d1_decay_gh200.yaml`. DAPO-v3-all and R3 are
  context. Exploratory label (exposed benchmark).
- **For.** Δ < 0 with the interval below 0 and every guardrail passing. Context reading: Δ
  below DAPO-v3-all's +0.85 and TPR above −10 pp.
- **Against.** Δ ≥ DAPO-v3-all's +0.85, or a failed TPR guardrail: the late decay does not
  undo what DAPO's last steps do.
- **Cost.** DAPO-v3-all trained 85 steps in 65 min; ≈ 5 GPU-h with start-up, 0.5 for the
  evaluation.
