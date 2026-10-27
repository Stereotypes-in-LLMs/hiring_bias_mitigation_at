# Counterfactual decision distillation: a frontier method from the survey

Started 2026-10-27 at the user's request, translated from Chinese: "Look at the other frontier
methods from the earlier survey, pick the one that looks most promising now and try it. Another
50 GPU-h." Branch `xinyang/decision-distill` (from main 9addb30). The comparison is
[`EVAL_STANDARD.md`](EVAL_STANDARD.md), unchanged; the reference arm is SFT-v3.

**Budget.** 50 GPU-h, separate from GRPO-improve's (163.8 spent of its 200). At most 3 jobs at
once, 3 retries. Costs under `HBM_COST_ARM=decision_distill`.

## 1. Why this method

The survey (`docs/FRONTIER_RL_SURVEY.md`, untracked; 147 methods, each checked by a source
agent and a sceptic) found three families still matched to the task. Family A,
consistency / privileged-context self-distillation with the model's own attribute-free
verdict as the target, was its first; the cheapest member it lists is "decision-token
frozen-teacher distillation (OPCD form, teacher SFT-v3 reading attr_free), 3–4 GPU-h"
(OPCD, arXiv 2602.12275; counterfactual logit pairing, arXiv 1809.10610).

What GRPO-improve and DAPO-improve found since the survey makes it the first choice now:

| finding (2026-10-25 to 27) | what it asks of the next method | this method |
|---|---|---|
| 80–98% of GRPO groups have no reward variance after step ~125 (explanation 1) | a training signal that does not depend on rollouts splitting | a dense loss on every row, no rollouts |
| the vote anchor contradicts SFT-v3's own attribute-free answer on 37% of rows; tipped-only data (R4) changed nothing (explanation 3) | a target that is the model's own judgement, not the anchor's | the target is SFT-v3 on the pair's attribute-free CV |
| the operating point drifts with the step size; every lr-decay run kept its guardrails (explanation 2) | hold the operating point | attribute-free anchor rows hold SFT-v3's attribute-free decisions; R3's lr decay |
| the evaluated quantity is whether a set's greedy decisions agree | train that quantity | the attributed decision is pulled to the attribute-free one |
| seed spread 0.4 pp (GRPO), 1.1 pp (DAPO) | several seeds from the start | ~3.5 GPU-h a seed: three seeds fit easily |

The survey's sceptic, recorded as a risk: the target is SFT-v3's soft verdict, which does not
sharpen, and the unstable sets sit on pairs where SFT-v3 is itself near its boundary; matching
a soft 0.5 may leave greedy signs free to differ. That is what the registration below can
refute.

## 2. Setup

- **Data** (`scripts/build_decision_distill_data.py` → `artifacts/semisynthetic-v3/distill-v3-all/`,
  report.json). grpo-v3-all's 9,140 attributed training rows (the rows every GRPO-improve arm
  trained on), each with its pair's attribute-free prompt from the round-3 pools, plus the
  attribute-free prompt as its own row for each of the 1,767 pairs: 10,907 rows. Read as text
  before training: the two prompts differ only in the injected attribute line, for example
  "+I am an active sikh and would need day offs for the major Sikh holidays." No benchmark
  candidate or job (checked against `data/benchmark/holdout_ids.json`).
- **Loss** (`mitigation/decision_distill.py`). After the op-margin-v2 prefix `{"decision": "`,
  KL(q ‖ p) over the first tokens of hire and reject: p the policy on the row's prompt, q
  SFT-v3 (the LoRA disabled, no_grad) on the pair's attribute-free prompt. No generation, no
  feedback tokens trained.
- **Training.** SFT-v3 merged, a fresh LoRA (r 32, α 64), lr 3e-5 with GRPO-improve's
  schedule (`warmup_stable_decay`, the last 89 of 340 steps decayed to 0), one epoch, one
  prompt per device and micro-step × 4 GPUs × 8 accumulation. Seeds 42, 43, 44
  (`configs/mitigation/distill/decision_distill_seed{42,43,44}.yaml`).
- **Evaluation.** The final adapter of each seed, declared now, by `evaluate_arm.sbatch` against
  SFT-v3 (`configs/audit/qwen3.5-9b_en_decision_distill_seed{…}_gh200.yaml`). The benchmark has
  been read many times in development; the verdicts are exploratory in that sense, as every
  GRPO-improve verdict was.

## 3. Pre-registration

- **Question.** Does decision distillation toward SFT-v3's attribute-free verdict lower
  unstable within-group sets at matched hire rate against SFT-v3, keeping every guardrail?
- **Control.** SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`; the
  comparison is op-margin-v2 compare-arms on 31,050 prompts and 2,700 sets. Context: GRPO +
  decay, five draws, mean −0.45 pp, none better (`GRPO_IMPROVE.md`, round 5).
- **For.** At least one seed better by the standard (interval below 0, every guardrail), and all
  three Δ < 0. Context reading: the three seeds' mean below GRPO + decay's −0.45 pp.
- **Against.** Any seed with Δ ≥ 0 or a failed guardrail; or a mean no better than −0.45 pp.
  The sceptic's mechanism would show as the attributed-row KL falling in training while the
  benchmark's unstable sets do not.
- **Stop conditions.** `memory_floor_gb` 12; a six-step smoke on `debug` before the real runs
  (`decision_distill_smoke.yaml`); a 2 h limit per run with `resume: true`.
- **Cost.** Smoke ≈ 0.6 GPU-h; per seed ≈ 1 h on 4 GPUs (≈ 3–4 GPU-h) plus 0.5 for the
  evaluation. About 12–14 GPU-h for three seeds.

## 4. Ledger

| # | Step | Job(s) | GPU-h | State |
|---|---|---|---|---|
| 1 | smoke | | | registered |
