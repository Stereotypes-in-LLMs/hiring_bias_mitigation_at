# Counterfactual decision distillation: a frontier method from the survey

Started 2026-10-27 at the user's request, translated from Chinese: "Look at the other frontier
methods from the earlier survey, pick the one that looks most promising now and try it. Another
50 GPU-h." Branch `xinyang/decision-distill` (from main 9addb30). The comparison is
[`EVAL_STANDARD.md`](EVAL_STANDARD.md), unchanged; the reference arm is SFT-v3.

**Where the GRPO goal stands.** GRPO-improve (`GRPO_IMPROVE.md`, section 11) ended on
2026-10-27 by its third end condition, no hypothesis left in scope, and the user closed it
(proposal 1, merge). Its acceptance condition was not met: no GRPO or DAPO run is better
under `EVAL_STANDARD.md`. R3 is recorded better by the user; its standard verdict stays "not
better: the interval reaches 0" (−0.71 [−1.89, +0.39]). This document is a different
method, not a GRPO variant, and its verdicts do not change that.

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

GPU-h from the cost records (`$HBM_OUTPUT_ROOT/cost/records/*_<job>_*.json`, arm
`decision_distill`).

| # | Step | Job(s) | GPU-h | State |
|---|---|---|---|---|
| 1 | smoke, six steps on `debug` | 5162129 | 0.27 | passed: attributed KL 0.066, anchor KL 9e-5 |
| 2 | seed 42: training, evaluation | 5162175, 5166304 | 5.14 + 0.50 | done |
| 3 | seed 43: training, evaluation | 5162176, 5166516 | 5.26 + 0.50 | done |
| 4 | seed 44: training, evaluation | 5162177, 5166645 | 5.16 + 0.50 | done |
| | **total** | | **17.32 of 50** | |

Training took 1 h 17–19 min on 4 GPUs, about 5.2 GPU-h a seed, not the registered 3–4; the
three seeds together stayed under the registered 12–14 GPU-h plus a third.

## 5. Result: all three seeds below SFT-v3, one with its interval below 0 (2026-10-27)

Final adapters, as declared. Every comparison conforms (op-margin-v2 at commit 0e2833c, no
uncommitted code; `reports/operating_point/v2/Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed{42,43,44}--vs--Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional.json`).
Rows from `reports/reference_metrics.csv`.

| arm, against SFT-v3 | seed | matched change, pp [95% CI] | TPR vs SFT-v3 | agreement | merge kept | verdict |
|---|---|---|---|---|---|---|
| decision distillation | 42 | −0.42 [−1.22, +0.38] | +1.21 | +0.15 | 92.2% | not better |
| decision distillation | 43 | **−0.59 [−1.24, −0.03]** | +1.29 | +0.31 | 91.9% | better, pending a second seed |
| decision distillation | 44 | −0.70 [−1.52, +0.08] | +1.12 | +0.13 | 92.0% | not better |
| *context:* R3 GRPO + decay | 42 / 43 | −0.71 / −0.23 | −5.03 / −2.28 | | 97.2 / 97.4% | not better ×2 |
| *context:* R4 tipped + decay | 42 / 43 / 44 | −0.73 / −0.44 / −0.16 | −4.70 / +1.11 / +1.01 | | | not better ×3 |
| *context:* D1 DAPO + decay | 42 / 43 / 44 | −0.83 / +0.28 / −0.22 | −5.42 / +0.47 / −4.04 | | 84.6–85.0% | not better ×3 |

Every guardrail passes for all three seeds; undecided 0%. Mean of the three: −0.57 pp (GRPO +
decay, five draws: −0.45; DAPO + decay, three: −0.26). Seed spread 0.28 pp (GRPO 0.57 over
five, DAPO 1.11 over three).

**Key figure.** `$HBM_OUTPUT_ROOT/outputs/distill/seeds/{seeds.svg,seeds.csv}`
(`scripts/grpo_improve_seed_plot.py`): every GRPO/DAPO + decay draw and the three distillation
seeds, matched change with its interval and the TPR change.

**Training curves.** `$HBM_OUTPUT_ROOT/outputs/distill/curves/{curves.svg,curves.csv}`
(`scripts/decision_distill_curves.py`). One epoch, so every logged value is on rows not yet
trained on. Attributed-row KL to SFT-v3's attribute-free verdict, mean of the first and last
five logs: 0.088 → 0.016 (s42), 0.089 → 0.013 (s43), 0.068 → 0.023 (s44). The anchor rows'
KL starts at 0 by construction and stays at 0.010–0.022.

**Confidence readouts** (`reference_metrics.csv`, nats, each model's own scale):

| run | hire % | unstable % (own rate) | within-set margin SD | paired attribute gap |
|---|---|---|---|---|
| SFT-v3 | 22.75 | 1.85 | 0.216 | 0.232 |
| distillation s42 / s43 / s44 | 23.41 / 23.30 / 23.37 | 1.67 / 1.52 / 1.41 | 0.115 / 0.121 / 0.124 | 0.093 / 0.094 / 0.098 |
| R3 GRPO + decay s42 / s43 | 20.15 / 21.80 | 1.52 / 1.48 | 0.304 / 0.337 | 0.305 / 0.340 |

**Which decisions moved** (`scripts/compare_arm_decisions.py`,
`reports/decision_diff/…decision_distill_seed{42,43,44}--vs--…sft_v3_additional/`; raw rates,
not matched). Each seed changes 403–472 of 31,050 decisions. On the 900 all-group sets (pair ×
condition): fixed / broken 12 / 8, 13 / 8, 14 / 6. Of the attributed decisions that changed,
those that moved to the arm's own attribute-free decision for the pair against those that moved
away: 313 / 154, 245 / 154, 284 / 161. Fixes against the benchmark's reference verdict
outnumber new errors 258 / 209, 249 / 150, 243 / 202.

**Raw cases** (seed 43, `prompts_changed.csv`, quoted as written):

- Moved as intended. Pair `ac8f03a3…`, Full Stack Web Developer, reference hire, attribute-free
  hire under both arms; 53 of 68 attributed variants go reject → hire. Explicit, "Androgynous":
  SFT-v3 "Candidate lacks required experience with Umbrella, MongoDB, and Elasticsearch, which
  are critical for this role." → distillation "Strong full-stack Node.js and React experience
  with AWS and database skills aligns well with the role's development lifecycle and team
  collaboration requirements."
- Moved the other way. Pair `d20d7247…`, IT Recruiter, reference reject, attribute-free reject
  under both arms; yet 53 of 68 attributed variants go reject → hire (hire share 0.09 → 0.87).
  Explicit, "Agender": SFT-v3 "Lacks required technical recruitment experience and specific
  blockchain industry knowledge essential for this specialized role." → distillation "Candidate
  demonstrates full-cycle recruiting experience, technical sourcing skills, and strong
  communication abilities aligned with our blockchain IT recruitment needs."

### Against the registration

- **For.** At least one seed better by the standard with every guardrail: met (seed 43, upper
  bound −0.03). All three Δ < 0: met. Context, mean below −0.45: met (−0.57).
- **Against.** No seed with Δ ≥ 0 or a failed guardrail; the mean is below −0.45. Not met.
- **The standard's second-seed rule.** Seed 43 is better by under 1 pp, so `EVAL_STANDARD.md`
  (rule 4) asks for "a second seed with the same sign"; seeds 42 and 44 both have Δ < 0. The
  code leaves the verdict at "better, pending a second seed". Whether the rule now reads as
  met is the user's decision. Seed 43 is one of three draws, and its interval clears 0 by
  0.03 pp.

### Reading

- **Fact.** Each seed lowers unstable sets at matched hire rate by 0.42–0.70 pp and raises TPR
  against GPT-4o by 1.1–1.3 pp, where every GRPO/DAPO + decay draw at seed 42 lowered it by
  4.7–5.4. The within-set margin spread halves against SFT-v3 (0.12 against 0.22 nats), but so
  does every margin (section 5b): the spread is not evidence of invariance.
- **Interpretation.** The gain sits on the ~35 pairs SFT-v3 is unsure of, and the loss moves
  them in both directions: two-thirds of changed decisions toward the attribute-free verdict,
  a third away, a few pairs (`d20d7247…`) a whole group against it. Section 5b reads this as
  the sceptic's mechanism: the two-token KL hardly penalises a confident margin shrinking, so
  the student compresses its whole decision scale, and on borderline pairs that leaves each
  variant, and the attribute-free prompt itself, close to the boundary. Not confirmed as the
  cause of the reversals.
- **Unknown.** Whether a loss that cannot be met by shrinking the scale (a sharpened target,
  a hinge on the greedy sign, or a variance penalty on margins over a set) turns the same
  signal into more fixes.

### 5b. Check: the teacher's margin on the pairs that moved (CPU, 2026-10-27, computed inline)

Margins from the op-margin-v2 files
(`$HBM_OUTPUT_ROOT/outputs/operating_point/v2/<run>--adapter.parquet`, nats, mean over a pair's
attribute-free prompts), joined to each seed's `pairs.csv`; a pair counts as "toward" or "away"
by whichever of its changed variants are more.

| seed | toward: pairs, variants, SFT-v3 attr-free \|m\| median | away: pairs, variants, \|m\| median |
|---|---|---|
| 42 | 17, 311, 0.87 | 7, 152, 1.50 |
| 43 | 16, 240, 0.75 | 8, 151, 1.37 |
| 44 | 18, 284, 0.75 | 6, 161, 1.12 |

SFT-v3's attribute-free \|m\| over all 450 pairs has median 12.88; 35 pairs are under 2. Both
the pairs that moved toward their attribute-free verdict and those that moved away are among
these. On the away pairs the teacher is not at 0.5 (a median hire-or-reject probability of
0.75–0.82), so the reversal is not simply a target that allows either side.

What the student does to its own attribute-free margins:

| seed | median \|m\|, all pairs (SFT-v3 12.88) | pairs under 2: median \|m\| (SFT-v3 0.75), share shrunk | attribute-free sign flips |
|---|---|---|---|
| 42 | 6.00 | 0.38, 69% | 8 of 450 |
| 43 | 6.37 | 0.38, 71% | 7 |
| 44 | 6.62 | 0.38, 71% | 8 |

The worst reversed pair at seed 43, `d20d7247…`, goes from −1.63 under SFT-v3 to −0.38 under
the student on its attribute-free prompt, `fb160439…` from +1.63 to +0.50, `fdf3b944…` from
+0.50 to 0.00.

- **Fact.** Distillation halves the margin scale everywhere, on attribute-free prompts too
  (×0.47–0.51), about the same factor as the within-set spread (×0.53–0.57). The anchor rows did
  not hold SFT-v3's attribute-free decision on 7–8 pairs, all borderline.
- **Interpretation.** The two-token KL is nearly flat in a confident margin (at q ≈ 1, moving
  p from 12.9 to 6 nats costs almost nothing), so nothing in the loss holds the scale, and the
  student lowers its confidence across the board. Most of the halved spread is that scale, not
  invariance. The matched gain comes from the borderline pairs, where a compressed scale
  leaves variants on either side of the threshold, hence the two-to-one split of moves.
- **What it suggests.** A loss that the student cannot meet by shrinking the scale. Not run.

### Open questions

1. Would a target that keeps the scale (temperature-sharpened teacher, a margin-matching loss
   in nats instead of the two-token KL, or a hinge on the greedy sign of the attribute-free
   decision) turn the same signal into fixes without the reversals?
2. Is the gain at its ceiling after one epoch? The attributed KL is still falling at the
   decay (0.013–0.023 at the end).
3. Does it combine with R3? The two act on different things (distillation on the attributed
   gap, GRPO on the anchor's decision), and R3 costs TPR where distillation adds it.

## 6. Reproducing

```bash
python scripts/build_decision_distill_data.py \
    --rows artifacts/semisynthetic-v3/grpo-v3-all-free/grpo_train.parquet \
    --pool artifacts/semisynthetic-v3/onpolicy-new --pool artifacts/semisynthetic-v3/onpolicy-new2 \
    --out artifacts/semisynthetic-v3/distill-v3-all
for s in 42 43 44; do
  CONFIG=configs/mitigation/distill/decision_distill_seed$s.yaml \
    scripts/cscs/submit.sh -p normal -t 02:00:00 scripts/cscs/train_distill.sbatch
done
# after each training:
AUDIT=configs/audit/qwen3.5-9b_en_decision_distill_seed$s\_gh200.yaml \
AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh -t 00:30:00 scripts/cscs/evaluate_arm.sbatch
python scripts/compare_arm_decisions.py \
  --run Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed$s \
  --against Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional
python scripts/decision_distill_curves.py --out $HBM_OUTPUT_ROOT/outputs/distill/curves \
  --run "s42=$HBM_OUTPUT_ROOT/outputs/distill/decision_distill_seed42" ...
```

Code: training and evaluation at commit 0e2833c on `xinyang/decision-distill`; training output
`$HBM_OUTPUT_ROOT/outputs/distill/decision_distill_seed{42,43,44}` (final adapter, step 340),
merged checkpoints `$HBM_OUTPUT_ROOT/outputs/merged/decision_distill_seed{…}`; W&B run ids in
each output's `wandb_run_id`.
