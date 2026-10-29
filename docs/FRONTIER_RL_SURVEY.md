# Frontier post-training methods for the GRPO problems: a survey (2026-10-22)

Written on 2026-10-22 in Chinese. Translated to English and brought up to date on 2026-10-29.
The analysis is kept as written. What ran after it is in
[Status on 2026-10-29](#status-on-2026-10-29). Statements that later work overtook stay where
they are, each followed by an *Updated 2026-10-29* note; errors found in the original are
marked *Corrected 2026-10-29*.

This page answers one question. GRPO-v3 and GRPO-v3-all exposed eight problems
(`docs/GRPO_V3.md`, and the user's table of observations of 2026-10-22). Which of the 2025–2026
frontier post-training methods are worth trying on this project, which are not, and why? It
was prompted by an NVIDIA engineer's suggestion to "try the most frontier methods".

The eight observations, numbered as in the user's table (reproduced with their evidence in
[`GDPO_CONSISTENCY_PLAN.md`](GDPO_CONSISTENCY_PLAN.md)):

1. Prompts SFT-v3 trained on carry almost no signal (GRPO-v3 on SFT-v3's own pairs: +0.74 pp).
2. The signal runs out within one epoch (groups with 16 identical rewards: 48% in steps 1–57,
   96% in steps 457–571).
3. Pre-selection by SFT-v3's error rate goes stale (87% of groups should split at SFT-v3's own p;
   52% did in the first 57 steps).
4. The anchor reward makes the model stricter than GPT-4o (TPR −10.11 pp; the guardrail fails).
5. Half the prompts train agreement with the anchor, not invariance (4,078 of 9,140 untipped).
6. The hire-rate monitor does not track the benchmark (checkpoint 525: projected 23.5%,
   audited 19.09%).
7. Decisions oscillate, and the verdict depends on the checkpoint (525 and 571 agree on 444 of
   450 attribute-free decisions; one is "better", the other not).
8. The format reward is constant (mean 0.200, group std 0.0006).

**Method.** Eight searches ran in parallel: dynamic prompt selection; consistency and
counterfactual rewards; privileged-teacher self-distillation; objectives and stability;
evaluation and checkpoint selection; NVIDIA's work; a scan for new 2026 methods; fairness. They
returned 96 entries, 89 methods after deduplication. Two independent agents then checked each
method. One checked that the source exists and that its mechanism and numbers match the paper.
The other argued the sceptic's side: is the method really useful here, with a binary decision,
no chain of thought, 16 rollouts, LoRA and a 45 GPU-h budget? It also gave the smallest
experiment that could be pre-registered. At the time of writing, 138 of the 147 methods had
both checks (useful here: maybe 56, no 82, yes 0) and 9 had only one, marked "unverified" in
the appendix. *Corrected 2026-10-29:* of those 9, 8 have neither check and 1 has only the
source check (`survey_raw.json`). Sources were checked by fetching the arXiv, blog or GitHub
page directly, and the numbers on those pages were compared. What could not be confirmed is
marked "partly accurate". The intermediate files are in
`/ritom/scratch/cscs/zhang/xw/tmp/survey/`.

## Status on 2026-10-29

**In brief.**
- *Fact.* Since this survey the project has run the cheapest member of its first family:
  decision-token distillation toward SFT-v3's own attribute-free decision, at three seeds. It
  has also run three of the seven tier-0 and tier-1 checks. GRPO-improve and DAPO-improve tested lr
  decay, a consistency reward and tipped-only data. Both closed on 2026-10-27 with no GRPO or
  DAPO run better than SFT-v3 under `EVAL_STANDARD.md`. The bar this survey set,
  DPO_V3_4880's −1.03 pp, did not replicate across seeds.
- *Interpretation.* The six structural facts and the "not worth a dedicated run" list still
  hold. Nothing run since contradicts them. The ranking held in one sense: family A gave the
  only gain that repeats across seeds, and that gain is small.
- *Unknown.* Whether a family-A loss that cannot be met by compressing the margin scale does
  better. None has been run.

**The bar.** At its own operating point DPO_V3_4880 gives −1.03, −0.09 and +0.20 pp over seeds
42, 43 and 44, a mean of −0.31 (sd 0.65) ([`DPO_V3_4880_SEEDS.md`](DPO_V3_4880_SEEDS.md)). At
SFT-v3's own hire rate, 22.51%, no recipe's seed mean is clearly ahead
([`FINAL_TABLE.md`](FINAL_TABLE.md), `reports/final/recipes.csv`):

| recipe | seeds | Δ at own point, mean (sd), pp | Δ at 22.51%, mean (sd), pp |
|---|---|---|---|
| DPO_V3_4880 | 3 | −0.31 (0.65) | −0.22 (0.63) |
| Decision distillation | 3 | −0.57 (0.14) | −0.28 (0.15) |
| GRPO + decay R4 | 3 | −0.44 (0.29) | −0.16 (0.26) |
| DAPO + decay D1 | 3 | −0.26 (0.56) | +0.13 (0.48) |

One comparison's interval is ±0.8 to ±1.2 pp ([`GRPO_IMPROVE.md`](GRPO_IMPROVE.md), section
11), about as wide as the seed spread. The standard's verdict is unchanged: own point, one seed
at a time, a second seed for a gain under 1 pp. `FINAL_TABLE.md` reads every arm beside it, at
22.51% and across seeds.

**Family A, consistency and self-distillation.**
- **Decision-token distillation, OPCD form** ([`DECISION_DISTILL.md`](DECISION_DISTILL.md)).
  KL from SFT-v3 (LoRA off) on the pair's attribute-free prompt to the policy on the row's
  prompt, over the first tokens of hire and reject after the op-margin-v2 prefix. 10,907 rows,
  one epoch, seeds 42–44. Own point: −0.42 [−1.22, +0.38], −0.59 [−1.24, −0.03] and
  −0.70 [−1.52, +0.08]; TPR +1.1 to +1.3 pp; every guardrail passes. The user recorded it as
  better than SFT-v3 on 2026-10-27, reading the second-seed rule as met. At 22.51% its mean is
  −0.28 (sd 0.15), and no seed's interval stays below 0. One more model costs 21.4 GPU-h on top
  of SFT-v3.
- **The sceptic's objection, tested.**
  - *Fact.* Distillation halves every margin, attribute-free ones too (median |m| 12.88 →
    6.00–6.62). A third of the changed attributed decisions move away from the attribute-free
    decision (`DECISION_DISTILL.md`, 5b).
  - *Fact.* Patching finds SFT-v3's attribute pathway in place, turned down to about a third
    ([`DISTILL_PATCHING.md`](DISTILL_PATCHING.md)).
  - *Fact.* A pure rescaling leaves the matched statistic at exactly 0. Every seed's gain
    survives against SFT-v3 rescaled by that seed's own factor, and the moved sets sit almost
    all on the 35 borderline pairs ([`SCALE_CONTROL.md`](SCALE_CONTROL.md)).
  - *Interpretation, not confirmed.* The gain is not the compression; it is a re-ranking of
    borderline pairs. The sceptic's "the soft target does not sharpen" shows up as the
    reversals, not as a null.
- **Layers 0–20 only**, seed 43: −0.28 [−0.96, +0.45], worse than the full adapter by
  +0.30 [+0.03, +0.62] ([`DISTILL_LAYERS.md`](DISTILL_LAYERS.md)).
- **GRPO on top of distillation**, round 6, seed 42: −0.17 [−1.07, +0.75], not below its start
  (−0.42). The share of zero-variance groups over steps 1–50 rose to 0.68, against R4's 0.59
  on the same batches (`GRPO_IMPROVE.md`, section 13). The user closed the round without
  seed 43.
- **Consistency reward**, GRPO-improve A2 (`GRPO_IMPROVE.md`, section 5). The target was
  static: SFT-v3's sampled attribute-free hire share for the pair. 200 steps. Mean proxy
  +0.05 pp against the control's −0.12: not better.
- **Not run:** BCT, the Debiasing-DPO variant, REM, SelfCI, OPSD, SDFT/SDPO, the KL gate.

**Family B, evaluation, selection and monitoring.**
- **lr cooldown.** GRPO-improve R3 decayed the lr linearly to 0 over the last 146 steps:
  −0.71 [−1.89, +0.39], TPR −5.03 pp against GRPO-v3-all's −10.11. The user recorded it as
  better (2026-10-26). It is the one change that kept every guardrail in 8 of 8 runs, and it is
  now the GRPO and DAPO default (`GRPO_IMPROVE.md`, sections 9 and 11). It held the operating
  point. It added no instability gain that survives seeds: GRPO + decay over five draws has a
  mean of −0.45, none better.
- **Tail averaging.** The equal average of the effective updates of the 7 checkpoints 425–571
  (job 5130000) gave TPR −3.0 pp and a proxy of −0.32 pp [−1.37, +0.67]. It centred the
  operating point and lost about half of the late checkpoints' gain (`GRPO_IMPROVE.md`,
  section 5; [`GRPO_CHECKPOINT_WEIGHT_AVERAGE_RESULTS.md`](GRPO_CHECKPOINT_WEIGHT_AVERAGE_RESULTS.md)).
- **The p(hire) monitor.** Built as the margin monitor: op-margin-v2 decision margins on 696
  held-out rows. Its calibration failed (job 5123830). It projected 5.0 to 6.4 pp too little TPR
  loss for DAPO-v3-all, GRPO-v3-all and checkpoint 525, mostly from the RJDB/Djinni source mix
  ([`GRPO_MONITOR8.md`](GRPO_MONITOR8.md)). A direct benchmark-dev monitor failed its replay: dev
  first fired at step 400, while the full benchmark had failed −10 pp at step 150
  ([`BENCHMARK_DEV_MONITOR.md`](BENCHMARK_DEV_MONITOR.md), job 5127818).
- **The checkpoint trajectory** (job 5127818). Full-benchmark TPR swings between −14.4 and
  +15.6 pp of SFT-v3's from one saved checkpoint to the next. The first −10 pp failure is at
  step 150 ([`GRPO_CHECKPOINT_TRAJECTORY.md`](GRPO_CHECKPOINT_TRAJECTORY.md)).
- **Resolution, observation 7.** No MDE was computed. Seed replicates confirm the reading: two
  GRPO seeds differ by 0.4 pp of mean proxy, DAPO D1 spreads 1.1 pp over three draws, and
  DPO_V3_4880's sd is 0.65 pp (`GRPO_IMPROVE.md`, section 5;
  [`DAPO_IMPROVE.md`](DAPO_IMPROVE.md), section 6; `DPO_V3_4880_SEEDS.md`).

**Family C, dynamic selection.**
- **DAPO-v3-all** ([`DAPO_V3.md`](DAPO_V3.md)). The dry-pool stop ended it at step 84 of ~570:
  +0.85 pp [−0.23, +2.12], TPR −10.10 pp, so the guardrail fails. The first-probe share of
  zero-variance groups reached 0.96 over steps 64–85.
- **NVIDIA's warning.** In DAPO-v3-all's last 13 steps the instability gain was lost while the
  pre-clip gradient norm rose from about 1 to 7.5, with fewer groups left per step and β = 0
  (`DAPO_IMPROVE.md`, section 1). Consistent with the warning. The pre-registered comparison
  (probe zero-variance share ≥ 10 pp above GRPO-v3-all's at equal prompt consumption) was not
  computed.
- **DAPO with R3's lr shape**, D1: −0.83, +0.28 and −0.22 over seeds 42–44, mean −0.26. D2 (D1
  with β 0.04): −0.34 (`DAPO_IMPROVE.md`).
- **Signal exhaustion is unfixed.** Zero-variance groups pass 0.8 by step ~125 in every GRPO
  arm (`GRPO_IMPROVE.md`, section 11).
- **Not run:** prompt replay, re-profiling, exact-logit selection, `scale_rewards='none'`.

**What the later work says about this page's claims.**
- Fact 4, the anchor's offset, now has a number. On 3,366 of the 9,140 training rows (37%) the
  vote anchor and SFT-v3's own attribute-free majority disagree, about equally often in each
  direction (`GRPO_IMPROVE.md`, explanation 3). Training on tipped rows only (R4) did not move
  the mean beyond seed noise.
- Fact 5, the consistency reward: A2 did not move the proxy.
- Fact 6, resolution: confirmed by the seed replicates above. Its third column says lr decay
  treats another noise source. That holds for the instability readout. But lr decay removed the
  TPR guardrail failure, which this page grouped under observation 7: the operating-point swing
  scales with the step size (`GRPO_IMPROVE.md`, R2-b, R3 and section 11).
- The Debiasing-DPO variant's premise, "a variant on the winner", no longer holds:
  DPO_V3_4880 is not the clear best arm.

**What is left if the work resumes.** This is an interpretation, not a plan.
- Family A with a loss that cannot be met by shrinking the margin scale: a temperature-sharpened
  teacher, a margin-matching loss in nats, or a hinge on the greedy sign of the attribute-free
  decision (`DECISION_DISTILL.md`, open question 1). BCT, this page's cheapest untried item, is
  the hard-label limit of the first. Cross-entropy toward the attribute-free decision rewards a
  confident margin, not a compressed one. Its anchor-target twin, the RFT control, was null
  (+0.31 pp, [`RFT_CONTROL.md`](RFT_CONTROL.md)).
- The Debiasing-DPO variant would need several seeds to be readable, since DPO_V3_4880's seed
  sd is 0.65 pp.
- The GRPO-family items wait on GRPO being reopened: REM, the on-policy consistency blocks of
  `GDPO_CONSISTENCY_PLAN.md`, prompt replay, re-profiling. The zero-variance problem they target
  is unchanged.
- The literature scan was not re-run on 2026-10-29; the appendix is the 2026-10-22 record. The
  installed versions this page names still hold: trl 1.15.0, with `trl.experimental.sdft`,
  `sdpo`, `distillation`, `iw_opd` and `async_distillation`, and `multi_objective_aggregation`
  in `GRPOConfig`; transformers 5.17.0, with `warmup_stable_decay` and `cosine_with_min_lr`;
  peft 0.21.2.

## The short answer (2026-10-22)

Most "frontier RL algorithms" are no-ops on this task or push the wrong way (the next section
says why). Three families match the causes and were not refuted by the sceptic:

1. **Consistency and privileged-teacher self-distillation, with the model's own attribute-free
   decision as the target** (BCT, OPCD, SDFT/SDPO, OPSD, SelfCI, REM, Co-rewarding). This is
   where the large labs' post-training converged in 2026: DeepSeek-V4, Qwen3.5-Omni and
   Nemotron 3 Ultra all do "RL per domain, then merge by on-policy distillation". It is also the
   dense form of the user's "consistency reward" proposal.
2. **Evaluation and checkpoint-selection methods with no training cost** (paired, clustered CIs
   with a minimum detectable effect; a teacher-forced p(hire) monitor; a pre-registered
   selection rule; tail checkpoint averaging). They address observations 6 and 7 directly, and
   observation 7 is probably not a training problem at all.
3. **Two cheap gate experiments for dynamic prompt selection**: re-profiling, to test whether
   observation 3 is staleness or selection bias; and prompt replay as a complement to DAPO.

The baseline to keep in mind: DPO_V3_4880 is already better than SFT-v3 by the standard
(−1.03 pp [−1.81, −0.28], all four guardrails pass, `docs/V3_NEW_DATA.md`). A new GRPO-family
arm means something only if it beats that number at matched hire rate and passes the
guardrails. Otherwise "frontier" is a −0.5 pp under a new name.

*Updated 2026-10-29:* that −1.03 is one seed of three. Seeds 43 and 44 gave
−0.09 [−0.88, +0.66] and +0.20 [−0.63, +1.11], a mean of −0.31 pp (sd 0.65)
(`DPO_V3_4880_SEEDS.md`). The bar now is under [Status on 2026-10-29](#status-on-2026-10-29).

## Why most frontier methods do not work here: six structural facts

| Fact | Evidence | What it rules out |
|---|---|---|
| The policy ratio is always 1 | One optimizer step per rollout (num_iterations=1); job 5116822 has clip_ratio 0.000 throughout | Every importance-ratio or clipping variant: GSPO, CISPO, SAPO, VESPO, LUSPO, GMPO, INTELLECT-2's two-sided clipping, clip-higher. In this configuration they are identical to REINFORCE. Only DAPO's two steps per rollout make them act |
| The audit decodes greedily, and the decision is one token | `docs/EVAL_STANDARD.md`: an unstable set is a greedy decision that flips across attribute values; flips happen only on borderline CVs whose attribute-free margin is near 0 | Methods that squeeze signal out of groups of 16 identical answers: BroRL's 512 rollouts, oversampling to 64/128 and filtering, RL-ZVP entropy shaping, Knapsack budget allocation. P(mixed) = 1 − p^N − (1 − p)^N says "all identical" is mostly undersampling, but a 1-in-64 dissent corresponds to a margin of about 4 nats, which attribute injection cannot move |
| Entropy sits in the feedback text, not in the decision | Mean token entropy in the logs is 0.42–0.44 throughout, with no collapse; the decision is 1 token, and the other ~38 tokens share its advantage | Entropy bonuses, adaptive entropy (Skywork-OR1), Clip-Cov / KL-Cov, top_entropy_quantile (its mechanism inverts on this format: the high-entropy tokens are the feedback's wording) |
| The TPR failure is a systematic offset of the anchor, not noise | On the 450 attribute-free benchmark CVs the anchor prompt rejects 79.1% (single draw) / 78.0% (greedy), GPT-4o 66.7% (`docs/SFT_V3_ANCHORS.md`, job 5071748). In training, agreement with the anchor rose 0.67 → 0.88 while the hire rate where GPT-4o hires fell 57.7% → 47.6% | Noise correction (noise-corrected GRPO, Cai et al.), soft targets (Brier against the vote share), Lagrangian constraints (Constrained GRPO: the constrained quantity was never violated on training rows), KL resets and mixup (ProRL, TR-DPO: the KL to SFT-v3 is the only term holding the hire rate, and a reset only releases it). The more the model learns to match the anchor, the more TPR falls |
| On tipped rows the consistency reward is byte-identical to the current reward | Tipped means SFT-v3's attribute-free majority agrees with the anchor (free_right_rate > 0.5, `generation/onpolicy.py`). On the 5,062 tipped rows "agree with your own attribute-free decision" equals "agree with the anchor"; only the 4,078 untipped rows differ | It changes the target of observations 4 and 5, but does not fix observation 2 (signal used up). A target taken from the current policy sharpens itself; one taken from frozen SFT-v3 copies the distribution SFT-v3 itself flips on at borderline CVs |
| Observation 7 is a resolution problem | 525 and 571 differ in only 6 of 450 attribute-free decisions. The two Δ (−1.02 / −0.56) differ by about 12 of 2,700 sets, while the 95% CI half-width is about 1 pp. TPR −10.11 against a limit of −10 | lr decay, EMA, checkpoint averaging and a second seed all treat other noise sources. Both numbers sit on a threshold: compute the paired, clustered SE and the minimum detectable effect (MDE) first |

One more hypothesis to test: **observation 3 may be selection bias, not staleness.** p comes
from the same samples that picked the rows (the winner's curse), so the "predicted 87% split"
is itself too high. A 1.5 GPU-h gate: re-sample SFT-v3 at k=16 on the 9,140 prompts. If the
predicted split rate is ≤ 60% (close to the measured 52%), it is selection bias, and the
staleness argument for picking prompts by the current policy fails. Only at ≥ 80% is it
staleness.

*Updated 2026-10-29:* not run as a gate. The paired probe ([`PAIRED_PROBE.md`](PAIRED_PROBE.md))
drew 16 fresh SFT-v3 samples on each of the 9,140 rows, the data the gate needs, but its report
does not compute the split rate.

## Methods by observation

| Observation | Frontier methods that fit | Sources | For this project | Cost |
|---|---|---|---|---|
| 2 The signal is used up in one epoch | Privileged-teacher self-distillation: the teacher is the same weights reading the attribute-free CV, the student reads the attributed CV, per-token reverse KL (all-identical groups get a gradient too) | OPCD 2602.12275; SDFT 2601.19897 (ICML 2026 spotlight); SDPO 2601.20802; OPSD 2601.18734; Thinking Machines OPD blog, 2025-10 | Maybe. Sceptic: the target is SFT-v3's own soft decision, which does not sharpen, and SFT-v3 itself flips on borderline CVs. The teacher sees less than the student, which is not the papers' "privilege" | Decision-token version 3–4 GPU-h; all-token version 26–30 GPU-h |
| 2, 3 Dynamic selection | DAPO dynamic sampling (in progress at the time; *updated 2026-10-29:* ran dry at step 85, not better, `DAPO_V3.md`); Prompt Replay as a complement to DAPO; Nemotron 3 Nano's profile → drop pass rate 1 → Gaussian curriculum → re-profile at a plateau; MoPPS / KGPS / ARCUS Bayesian selection (no extra inference); PCL's value model can be swapped for the decision-token probability | 2603.21177; 2512.20848; 2507.04632 / 2607.27610 / 2609.38018; PCL (Gao et al., Meta/Cornell, 2025-10) | Maybe (replay, the re-profile gate, exact-logit selection). No for pool exhaustion: with 96% of groups identical, selection can only recover the remaining 4–10% | Gate 1.5 GPU-h; selector 0 extra GPU |
| 4, 5 The target is stricter than GPT-4o / half the prompts teach "match the anchor" | Consistency training, BCT (SFT: attributed prompt → own attribute-free answer); a cross-prompt consistency reward (Co-rewarding's cross view); REM, paired-arm expectation matching; SelfCI, a two-teacher product; Debiasing-DPO | 2510.27062 (DeepMind); Co-rewarding (ICLR 2026); 2610.22942 (Meta, 2026-10-16); 2605.20258; 2604.02585 (COLM 2026) | Maybe. BCT is cheapest, but the RFT control (+0.31 pp) is its anchor-target version, and it was null. REM has zero gradient on all-identical groups. Debiasing-DPO is this repository's DPO recipe (sigmoid + 0.2 NLL); the one change not tried is taking chosen from the model's own attribute-free decision | BCT 5–7 GPU-h; consistency-reward GRPO ~24; REM 35–40; Debiasing-DPO variant ~5 |
| 6 The monitor is off | Teacher-forced p(hire): one forward pass per monitor prompt reads the decision-token probability → expected flip rate, boundary mass, hire rate, a TPR proxy, KL; a control-pair noise floor | Counterfactual logit pairing, Garg et al., AIES 2019 (1809.10610); ICE-Guard / HALF 2510.12217 (checked: maybe, but it is a measurement, not a training change; at 128–440 pairs its SE is 1.5–3 pp) | Maybe. It is a concrete form of the user's "monitor the flip rate directly". The TPR proxy needs one GPT-4o labelling of about 1,000 control prompts | Replay over 20 checkpoints 1–2 GPU-h; 2–5 GPU-min every 25 steps |
| 7 The verdict swings, and the conclusion depends on the checkpoint | Paired, clustered CI + MDE; pre-registered selection (disjoint dev set); tail checkpoint averaging (LAWA); the rule "audit where the rolling mean of frac_reward_zero_std first exceeds 0.9"; lr cooldown (WSD) | 2411.00640 (Anthropic); 2504.07086 (COLM 2025); 2502.06761; 2606.30789; transformers 5.17's warmup_stable_decay | Maybe, but do tier 0 first: observation 7 is most likely a detection-limit problem. Averaging and lr decay are alternatives (Ajroldi: tail averaging at constant lr is about equivalent to annealing) | 0 GPU-h (CI / MDE); averaging 0.5 GPU-h each |
| 8 The format reward is constant | GDPO (NVIDIA), decoupled normalisation; Constrained GRPO's standardized-advantage scalarisation | 2601.05242; 2602.05863 | No. Every candidate reward is a deterministic function of the same binary token, so GDPO has nothing to act on. TRL 1.15 already has multi_objective_aggregation='normalize_then_sum', which matters only once a second, non-constant reward exists | 0 |

## Candidates in the three families

### A. Consistency and privileged-teacher self-distillation (target: the model's own attribute-free decision)

What they share: the attribute-free prompt stored with each pair (the pool's
`prompts.parquet`, kind='attr_free', joined on pair_id) is the privileged context, and teacher
and student are the same weights. They differ in three ways. The target is either the
log-prob difference of the sampled token (OPD) or a full-vocabulary KL (OPCD / OPSD). The
teacher is frozen (SFT-v3, adapter off), an EMA or live. And the KL runs in one direction or the
other.

*Updated 2026-10-29:* the decision-token OPCD form ran at three seeds
(`DECISION_DISTILL.md`); its result is under [Status on 2026-10-29](#status-on-2026-10-29).

| Method | Mechanism in one sentence | For this project | Sceptic |
|---|---|---|---|
| **OPCD** (Microsoft, 2602.12275, 2026-02) | The student samples without the context; a frozen teacher has the context; reverse KL over the student's top-k | The paper finds a frozen teacher better than "being your own teacher" (Sokoban 53.9 vs 18.8). On a safety-classification task, on-policy beats off-policy context distillation, the latter being SFT on attribute-free outputs. A version that distils only the decision token costs 3–4 GPU-h | The fixed point is SFT-v3's soft decision on the attribute-free CV, which does not sharpen. The unstable sets are almost all on pairs at SFT-v3's own boundary |
| **SDFT / SDPO** (MIT / ETH, 2601.19897, ICML 2026 spotlight; 2601.20802) | SDFT: the teacher is the same network plus privileged examples, EMA weights, full-vocabulary KL. SDPO: (1−w)·policy loss + w·distillation, with a teacher that sees the prompt plus successful samples from the group or feedback | TRL 1.15 ships `trl.experimental.sdft` / `sdpo` (teacher_model_kind ∈ {base, ema, live}, distillation_mode ∈ {sampled_token, topk_logits, full_logits}, distillation_weight), plus distillation, iw_opd and async_distillation. But they inherit from _BaseTrainer, not GRPOTrainer: this project's weight sync, merge-retention check and decision monitor would have to be ported as callbacks. The teacher's prompt template has to replace the student's prompt, not be appended to it | "Dense in tokens is not dense in information": the papers' teacher sees a solution or feedback, while here the teacher sees less than the student |
| **Thinking Machines OPD** (blog, 2025-10) | advantage_t = −(log π_S − log π_T) per token, discount 0, reusing RL's importance-sampling loss as is | One extra no-grad forward pass ([x_free; y], adapter off) in a subclass of the existing GRPOTrainer; +15–25% step time | Conflicts with DAPO: the all-identical groups DAPO drops are exactly where it claims to act |
| **OPSD** (UCLA / Meta, 2601.18734, 2026-01) | The teacher is fixed at the initial policy plus privileged information; forward KL over the full vocabulary; G=1 | 8–12 GPU-h per epoch | The same greedy-audit argument |
| **SelfCI** (KAIST, 2605.20258, 2026-05) | The target is exactly "the next-token distribution does not change when disallowed information is added". It finds a plain information-removed teacher underdetermined and uses the product of the information-removed teacher and the task teacher | The closest problem form to this project's (LoRA r32, 7–8B, a GRPO scalar reward as the baseline). Two teacher forward passes, 32–36 GPU-h | Same as above |
| **Gate, to run first**: Zou et al. 2610.22512 (2026-10-15) | Before training, compute the sampled-token reverse KL and a "wrong-step KL contrast" to predict whether privileged information will help | 1–2 GPU-h: the decision token's KL(attr‖free) on SFT-v3 and checkpoints 525 / 571. If the median KL on all-identical groups is small, this family has no signal, and none of it runs | Tipped is already defined by a sampled estimate of this KL, so the gate risks being circular |

| Paired / consistency reward | Mechanism | Notes | Sceptic |
|---|---|---|---|
| **BCT** (DeepMind, 2510.27062, 2025-10) | The current model generates the target on the clean prompt; plain SFT on the prompt with the cue | The cheapest (5–7 GPU-h), and the hire rate is pinned to the model's own attribute-free hire rate by construction. The target can be SFT-v3's 8-vote attribute-free majority already stored in the pool | The RFT control (job 5066776, +0.31 pp) is its "anchor target" version, and it was null. So this is the deciding experiment for "contrast vs imitation", not a new method |
| **Cross-prompt consistency reward** (Co-rewarding, ICLR 2026) | Score one view by the other view's majority vote. This avoids the collapse of single-view self-consistency, where the majority pseudo-label's accuracy can fall to ~0 | The concrete form of the user's proposal for observation 4. A TRL reward function can read dataset columns, so adding a neutral_decision column is enough | Identical to the current reward on tipped rows. It differs only on the 4,078 untipped rows, and there the target is a noisier self-target |
| **REM** (Meta, 2610.22942, 2026-10-16) | One GRPO group per arm, plus a correction term −α(μ−μ′)(r_i−r̄_{−i}) that matches the expected rewards | The closest published analogue. The gap in hire indicators is exactly the quantity observation 6 wants to monitor | With a 0/1 decision the correction is identically 0 on all-identical groups. Zero gradient on deterministic flips (neutral all hire, attributed all reject) |
| **Debiasing-DPO** (Stanford, 2604.02585, COLM 2026) | chosen = the model's own answer without the spurious context, rejected = its answer with it, plus an SFT anchor | This repository's DPO recipe already has this structure (loss_type [sigmoid, sft], weights 1.0 / 0.2). The one change not tried: take chosen's decision from the model's own attribute-free decision instead of the anchor. DPO_V3_4880 is the best arm now, so this is a ~5 GPU-h variant on the winner. *Updated 2026-10-29:* not the clear best arm any more (three-seed mean −0.31 pp, sd 0.65) | Duplicates DPO_V3_4880; new only in the target |
| PA-GRPO (ACL 2026, 2603.21016): maybe. EIS-GRPO / J4R (Salesforce, 2505.13346): no. J1 (Meta, 2505.10320): no. BCT-Chua (2403.05518): maybe. Causal-consistency DPO (Hüyük, ICLR 2025, 2410.03767): maybe | Put every counterfactual variant of an item in one advantage group, or reward only when both orders are right | Structurally the same as "the counterfactual set inside a group". Needs a subclass that changes the advantage grouping | PA-GRPO's modal-consistency reward (±1) overwhelms the 0/+1 anchor reward and, on CVs anchored hire where the policy leans reject, pushes toward reject. EIS's new signal (a subgroup all identical while a sibling variant splits) is nearly empty on this data. J1's AND reward changes prompt weights, not what is learned. Consistency-DPO's relabelling is a no-op on the existing DPO pairs, which are built only on tipped rows |

### B. Evaluation, selection and monitoring (no training cost)

| Method | What it does | Notes |
|---|---|---|
| Paired, clustered CI + MDE (Miller 2024, 2411.00640) | The paired difference per set, clustered by candidate; compute the minimum detectable effect before training | The matched_comparison in `scripts/operating_point_control.py` is already a paired, candidate-clustered bootstrap. What is missing: reporting SE, MDE80, MDE90 and the design effect, and re-estimating the matched cut inside each bootstrap draw. 0 GPU-h. The result would go into EVAL_STANDARD |
| Teacher-forced p(hire) monitor | After the fixed prefix `{"decision": "`, read P(hire)/(P(hire)+P(reject)) in one forward pass | Replaces the "projected hire rate" band (23.5% against an actual 19.09%). Replay it over GRPO-v3-all's 20 checkpoints to check whether it predicts the audit. The expected flip rate must follow the greedy definition (the sign of the margin), not independent draws |
| Pre-registered selection (2504.07086) | A candidate set (every 25 steps, the last step, the tail average), a disjoint dev set, an eligibility gate, an objective, one audit only | Note: 571 was pre-registered in the first place, and 525 was exploratory. Observation 7 was not caused by a missing rule |
| Tail checkpoint averaging (LAWA; Ajroldi 2025, 2502.06761) | Average the merged deltas (B·A, not A and B separately) of the last k adapters | A constant lr is exactly where averaging gains most. It can be applied after the fact to 475–571, and merge retention ≥ 80% must be rechecked. An alternative to lr decay. Sceptic: the move toward reject is a drift, not zero-mean noise, and averaging only gets back to about step 523 |
| Zero-variance rule (Predictable GRPO, 2606.30789) | "Audit where the 50-step rolling mean of frac_reward_zero_std first exceeds 0.9" | On 5116822 it gives step 231 → checkpoint-250. Cheap, and can be pre-registered |
| LITI / WiSE-FT interpolation (Rewarded soups, NeurIPS 2023; WARP 2406.16768) | θ = θ_SFT-v3 + η·Δ, with η chosen on dev to match the hire rate or TPR | Makes the TPR guardrail pass by construction. Largely duplicates op-margin-v2 |
| lr cooldown | transformers 5.17 has warmup_stable_decay and cosine_with_min_lr | Tier 0 first: if the CI of the 525 vs 571 Δ contains 0, observation 7 is a reporting problem, not a schedule problem |

*Updated 2026-10-29:* the monitor was built and failed its calibration; tail averaging and the
lr cooldown were run. See [Status on 2026-10-29](#status-on-2026-10-29).

### C. Dynamic selection (observations 2 and 3): complements to DAPO

- **Nemotron 3 Nano's recipe** (2512.20848): profile with the SFT checkpoint → drop pass rate
  1.0 → a Gaussian curriculum on pass rate → at a plateau, re-profile with the best RL
  checkpoint. `build_grpo_prompts.py` already does the first three; re-profiling is new. Stage 0
  is the observation-3 gate above.
- **Prompt Replay** (2603.21177): keep recently split prompts in a priority queue instead of
  DAPO's RefillOrder shuffle. A direct remedy for DAPO's "stop when fill < 0.5". Note: prompts
  that still split late in training are mostly ones whose anchor is itself close to a tie (≤ 6
  of 8 votes), so admit only those with a vote margin ≥ 7/8.
- **Exact-logit selection** (PCL's value model swapped for the decision-token probability).
  Without a chain of thought, p_θ(anchor | prompt, prefix) is the exact expected reward, with
  far lower variance than 16 Bernoulli draws; sweep the whole pool with vLLM max_tokens=1. But
  "p ≈ 0.5" does not separate tipped rows (p_free ≈ 1) from borderline CVs (p_free ≈ 0.5); read
  it together with the attribute-free p.
- **Class-balanced dynamic sampling** (BD-GRPO, 2605.29076; checked: no, its mechanism is for
  class-imbalanced data, while here the two sides are equal and DAPO already balances them).
  The closest published analogue: GRPO for text classification on a small Qwen3. It finds that
  zero-variance filtering without class balance collapses to the majority class, which is the
  "drift toward reject" here. It supports DAPO's strict balance.
- **NVIDIA's counter-evidence.** AceReason-Nemotron and Nemotron-Cascade (2512.13607) both find
  that several steps per rollout with clip-higher collapse entropy within 100–150 steps, and
  that strict single-step on-policy training is more stable. DAPO-v3 uses two steps and 0.28. A
  prediction that can be pre-registered: if DAPO's dapo/probe_zero_variance is ≥ 10 pp above
  GRPO-v3-all's at the same prompt consumption, collapse is accelerating.
- **scale_rewards='none' / Dr. GRPO** (Bereket & Leskovec, 2508.11800). Within-group std
  normalisation gives the lone dissenter in a 15/1 group |A| ≈ 3.9, so one or two lone
  dissenters set the direction of the last 100 steps. One switch, but it changes the relative
  weight of the hire and reject sides (Σk(16−k) vs Σ√(k(16−k))).

*Updated 2026-10-29:* DAPO-v3-all ran dry at step 85 and was not better; DAPO with R3's lr
decay (D1) gave −0.83, +0.28 and −0.22 over three seeds. Prompt replay, re-profiling,
exact-logit selection and `scale_rewards='none'` were not run. See
[Status on 2026-10-29](#status-on-2026-10-29).

### D. NVIDIA's work: what fits

| NVIDIA method | Verdict |
|---|---|
| ProRL / ProRL v2 (2505.24864; 2507.12507): a KL trust region with periodic reference resets | No. Here the KL does not "dominate and freeze learning" (k3 KL 0.016–0.035 nats/token throughout), and the KL to SFT-v3 is the only term holding the hire rate; a reset would only release TPR |
| BroRL (2510.01180, ICML 2026): 512 rollouts per prompt | No. Its mechanism is coverage in a large output space. Here the reward sits on one binary token, and the greedy audit cannot see a 1-in-64 dissent. A 0.3–0.5 GPU-h gate could confirm this |
| GDPO (2601.05242): decoupled normalisation of several rewards | No (confirms observation 8). TRL 1.15 already has multi_objective_aggregation='normalize_then_sum'; revisit once there is a second, non-constant reward |
| DLER: batch-level reward normalisation, dynamic sampling, clip-higher, a length penalty | The length penalty is irrelevant. Batch-level normalisation is one switch, scale_rewards='batch'. DAPO already has the rest |
| RLBFF (2509.21319, ICLR 2026) / GenRM: a principle reward on the feedback field | No. The lexical detector hardly fires on the 31,050 audit responses, and the feedback does not mention the attribute. It also breaks the design rule of keeping models out of the reward |
| QeRL (2510.11696): quantisation noise as exploration | No. The saturation is not a lack of exploration |
| RLP (2510.01265, ICLR 2026) | Not applicable: no pre-training stage, no chain of thought |
| Nemotron 3 Nano / Super / Ultra recipes: profiling, multi-environment joint training, GenRM | Multi-environment training prevents cross-domain forgetting, and the TPR failure is not forgetting. For profiling and re-profiling see C |
| NeMo-RL (v0.4–0.7) importance-sampling corrections (TIS / ICE-POP / seq-mask-tis) | TRL 1.15 has the same kind of switch, on by default. The measured log-prob gap between vLLM and the trainer is 0.012 nats/token, a ratio of 0.999: nothing to correct |
| **NVIDIA's own direction**: NeMo-RL v0.7 added multi-teacher on-policy distillation (MOPD) for Nemotron 3 Ultra | Consistent with family A: NVIDIA too is moving toward on-policy distillation |

### E. Post-training recipes in the 2026 model reports

Qwen3.5, this project's base, has no technical report of its own. The Qwen line's public recipe
in 2026 trains domain experts with GSPO (= TRL importance_sampling_level='sequence') and then
merges them by on-policy distillation (Qwen3.5-Omni, 2604.15804). DeepSeek-V4 (2606.19348)
trains GRPO experts and merges them by reverse-KL on-policy distillation, which replaces the
mixed RL stage. GLM-5 has the IcePop mask (checked: a mismatch correction; the mismatch here is
about 0). What they share: RL sets the direction, distillation sets the magnitude. The analogue
for this project (checked: no) uses a mixture of late checkpoints as the OPD teacher. For one
binary token the optimum of the reverse KL is the mixture itself: the student converges to that
probability and is not more stable.

## Recommended experiment sequence (by cost; each step can be pre-registered)

The status column was added on 2026-10-29; the evidence is under
[Status on 2026-10-29](#status-on-2026-10-29).

| Tier | Experiment | GPU-h | Decides | Status, 2026-10-29 |
|---|---|---|---|---|
| 0 | Paired, clustered SE, MDE80 / 90 and the design effect on the existing op-margin-v2 files; the CI of the 525 vs 571 Δ | 0 | Whether observation 7 is a detection limit; how many pp a "win" needs from now on | Not run as such; seed replicates settled the question |
| 0 | Replay 5116822's log: the step where the frac_reward_zero_std rule fires; per-group reward by tipped / untipped and by 1–2 / 3–13 / 14–15 | 0 | Whether lone-dissenter advantages dominate late training; candidates for a selection rule | The zero-variance share was replayed (`GRPO_IMPROVE.md`, section 2); no selection rule was adopted |
| 0 | Load the adapters of checkpoints 475–571 and compute the pairwise distances of the merged deltas | 0 (CPU) | Whether tail averaging is worth one audit | Skipped: the average was built and scored directly |
| 1 | Re-sample SFT-v3 at k=16 on the 9,140 prompts | 1.5 | Observation 3: selection bias vs staleness (thresholds ≤ 60% / ≥ 80% written in advance) | Not run as a gate; the paired probe's samples could answer it |
| 1 | The decision token's KL(attr‖free) for SFT-v3 and checkpoints 525 and 571, on the training set and the 256 monitor rows | 1–2 | Whether family A has any signal (the median KL on all-identical groups) | Not run; distillation ran directly |
| 1 | The teacher-forced p(hire) monitor replayed over 20 checkpoints; one GPT-4o labelling of ~1,000 control prompts for a TPR proxy | 1–2 | Whether the new monitor predicts the audit; a dev objective for the selection rule | Built as the margin monitor; its calibration failed (job 5123830). The benchmark-dev monitor failed its replay |
| 1 | A tail-averaged adapter (475–571), merged and audited, labelled exploratory | 0.5–0.7 | The cheapest treatment of observation 7 | Run over the 7 checkpoints 425–571 (job 5130000): it centred TPR and lost about half the gain |
| 2 | BCT: SFT on the 9,140 attributed prompts toward SFT-v3's own 8-vote attribute-free majority JSON; control: the RFT control | 5–7 | Separates "contrast vs imitation" and "anchor vs own decision" in one run | Not run |
| 2 | Debiasing-DPO variant: DPO_V3_4880's recipe unchanged, chosen's decision replaced by the model's own attribute-free decision | ~5 | The effect of the target, on the best arm at the time | Not run; it would now need several seeds |
| 2 | Decision-token frozen-teacher distillation (OPCD form, teacher SFT-v3 reading attr_free) | 3–4 | The smallest form of family A | Run at 3 seeds: mean −0.57 pp (sd 0.14), recorded better by the user |
| 2 | Consistency-reward GRPO (the user's proposal) with DAPO's mechanics | ~24 | Runs if tier 1's KL gate passes | Run as A2, without DAPO, 200 steps: not better |
| 3 | REM / paired-group GRPO; a second seed | 35–40 / 24 | Only after a positive result above | REM not run; seeds 43 and 44 run for every open recipe |

**Not worth a dedicated run** (one-line reasons): GSPO / CISPO / SAPO / VESPO / LUSPO / GMPO /
two-sided clipping (no-ops when the ratio is 1); entropy bonuses / Clip-Cov / top-entropy (the
entropy is in the feedback); BroRL / oversampling / Knapsack / RL-ZVP (the greedy audit cannot
see rare dissent); ProRL resets / TR-DPO mixup (they release the one term holding TPR);
noise-corrected GRPO (identical to the original advantage under a binary reward); Lagrangian
constraints (the constraint was never violated on training rows); GDPO (no second non-constant
reward); Nemotron-Cascade's epoch-level filtering (nothing to act on in one epoch); QeRL / RLP;
TTRL's single-view majority vote (it would reward the attribute-induced flip itself); RLBFF /
GenRM on the feedback (the feedback does not mention the attribute); RELEX trajectory
extrapolation (5116822's adapter trajectory is not a straight line); fp16 / fp32 precision
fixes (measured mismatch 0.012 nats/token). *Updated 2026-10-29:* nothing run since
contradicts this list.

## Verification status and limits

- In the first round, verification failed for 21 of the 89 methods (a quota interruption), and
  they were re-run. A gap critic named 8 gaps; the follow-up search added 58 methods, checked
  the same two ways, for 147 methods in the appendix (138 with both checks, 9 with one).
  *Corrected 2026-10-29 from `survey_raw.json`:* of those 9, 8 have neither check and 1 has the
  source check only. Six of the 9 are gap methods and three are from the first round (rows 108,
  118 and 138), so the first round's re-run was not complete either. Several have a fully
  checked near-duplicate: REM in rows 22, 23 and 98, FRM in row 119, RL-ZVP in row 51, concept
  editing in rows 16 and 72.
- Sources were checked mainly by fetching arXiv, blog and GitHub pages directly. Most agents ran
  out of WebSearch budget partway, so acceptance at a venue follows the arXiv comments or
  OpenReview and was not cross-checked.
- The critic's 8 gaps and what the follow-up found:
  - Counterfactual logit pairing, CLP (Garg 2019, an |Δlogit| penalty on the decision token):
    maybe; the oldest and most direct form in family A.
  - A second reward on the feedback field (FRM, a DeBERTa neutrality classifier): no. The
    feedback does not mention the attribute; attribute_mention_rate is 0.
  - Auditing the anchor against GPT-4o first, as a confusion matrix: already in
    `reports/teacher_anchor/`; SFT-v3 already sits at the teacher's operating point.
  - Synthesising new prompts after saturation (AZR, variational problem synthesis): no; what is
    missing is uncertainty, not prompts.
  - A decision-token logit_bias as a hire-rate dial: already op-margin-v2's standard.
  - Concept erasure and activation consistency (ACE, LEACE, ACT): maybe, but they remove only
    the group-mean shift, and borderline CVs do not gain.
  - Qwen3.5's own recipe: blog-level, no testable method.
  - A sampling-noise floor, a paired null, non-inferiority margins and other evaluation
    statistics: mostly "a different estimator", duplicating the current standard.
  - Still not covered: FARO (2602.07799) and Nemotron Nano 2's RL recipe. The resume-screening
    paper ("adversarial contrastive tuning", Information and Software Technology 193, 108040)
    is appendix row 88, but it is closed access (ScienceDirect returned 403), so it was judged
    from its metadata and the related arXiv work.
- A side finding of the follow-up worth recording: a sceptic cited the finished DAPO-v3-all's
  `dynamic_sampling.jsonl` (job 5119438) to argue that the premise of selecting prompts by the
  current policy "came up empty" in that run (9,248 groups probed). For that conclusion the
  formal result in `docs/DAPO_V3.md` is the reference.
- The critic's tally: none of the 89 methods was judged "yes", and observation 8 (the constant
  format reward) has no workable method. The ranking could only be among the "maybe"s.
- The sceptic's verdicts are an agent's judgement, not experiments. Each "no" comes with a
  0–2 GPU-h gate that data can overturn.
- This is a hand-written survey page, not an experiment card. It was written on the
  `xinyang/dapo` worktree and stayed uncommitted until 2026-10-29, when it was translated,
  brought up to date and committed. The appendix was generated with Chinese labels by
  `build_report.py` from `survey_raw.json`, both in the intermediate directory above. The
  English labels and two source IDs were changed by hand: in rows 11 and 88 the generator had
  taken a DOI fragment for an arXiv ID.

## Appendix: all 147 methods

Sorted by the priority the search agent gave (5 = try first). A method marked [gap] was added by
the follow-up search. "Source check" comes from the source agent; "Useful here?" and the
objection come from the sceptic. Each cell is cut to a fixed length as generated; the full text
is in `survey_raw.json`.

| # | Method | Source | Date | Observations | Source check | Useful here? | Sceptic's main objection |
|---|---|---|---|---|---|---|---|
| 1 | Activation Consistency Training (ACT) on counterfactual pairs: MSE between the policy's hidden states on the a… [gap] | Mitigating Adaptive Attacks against Reasoning Models with Activation Consistency Training… (2605.28467) | 2026-05 ;… | 2,3,4,5 | exists, partly accurate | maybe | The decision metric is greedy-argmax instability (docs/EVAL_STANDARD.md: greedy decoding, thinking off, merged weights), and the project's own evidence (scripts… |
| 2 | Anchor-preserving variational prompt synthesis gated by the CURRENT policy's split rate (SvS recipe applied to… [gap] | Beyond Pass@1: Self-Play with Variational Problem Synthesis Sustains RLVR (2508.14029) | 2025-08-19… | 2,3,5 | unverified | unverified | — |
| 3 | Bayesian pre-rollout prompt selection (MoPPS -> KGPS / ARCUS, with ThinkPrior-style anchor prior) | Can Prompt Difficulty be Online Predicted for Accelerating RL Finetuning of Reasoning Mode… (2507.04632) | 2025-07 (M… | 2,3 | exists, mechanism accurate | no | The method estimates a quantity this project can read exactly, and cannot track the one it claims to repair. (1) With no chain-of-thought, a prompt's "pass rate… |
| 4 | BroRL-style broadened rollouts (many more generations per prompt, fewer prompts per step) | BroRL: Scaling Reinforcement Learning via Broadened Exploration (2510.01180) | 2025-10 | 2,3 | exists, mechanism accurate | no | BroRL's mechanism needs a large answer space in which correct trajectories go unsampled; here the answer space is two verdicts and the audit is greedy (configs/… |
| 5 | Consistency Training (BCT): supervised fine-tuning on the model's own attribute-free response as the target un… | Consistency Training Helps Stop Sycophancy and Jailbreaks (2510.27062) | 2025-10 | 1,2,3,4,5,6 | exists, partly accurate | maybe | The project has already run the closest analog of BCT and it did not move the number. The RFT control (docs/RFT_CONTROL.md, job 5066776) was one SFT epoch on a… |
| 6 | Counterfactual Logit Pairing (CLP) on the decision token: \|logit_hire(x_attr) - logit_hire(x_free)\| + CE/KL… [gap] | Counterfactual Fairness in Text Classification through Robustness (introduces counterfactu… (1809.10610) | 2018-09 (a… | 2,3,4,5,6,8 | exists, mechanism accurate | maybe | The headline metric is a sign-crossing count under GREEDY decoding (audit temperature 0.0, `eval/metrics.py`, `reference_metrics.py::unstable_within_group`), an… |
| 7 | Counterfactual-context self-distillation (SDFT form): teacher = same weights on the attribute-free prompt, stu… | Self-Distillation Enables Continual Learning (2601.19897) | 2026-01 (v… | 2,4,5,6 | exists, partly accurate | no | The audit is greedy and set-level (sign of the decision margin must agree across attribute values), and SFT-v3's instability lives almost entirely on pairs wher… |
| 8 | Cross-prompt consistency reward: attributed rollouts rewarded for agreeing with the model's own attribute-free… | Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models (… (2508.00410) | 2025-08 (v… | 2,3,4,5 | exists, partly accurate | maybe | The method adds no new signal where the signal lives and its genuinely new part is a noisy self-sharpening target. `tipped` is defined in generation/onpolicy.py… |
| 9 | Decision-token logit bias as an exact hire-rate dial (vLLM SamplingParams.logit_bias) [gap] | Evaluating and Recalibrating Probabilistic Classifiers from One-Bit Threshold Queries (v2;… (2609.05125) | 2026-09 (v… | 4,6,7 | exists, partly accurate | no | The project already does this, more exactly, and has fixed it as THE decision metric: docs/EVAL_STANDARD.md and /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigat… |
| 10 | Drop group std normalization (scale_rewards='none' / Dr. GRPO) so stochastic (borderline) anchors do not drive… | Uncalibrated Reasoning: GRPO Induces Overconfidence for Stochastic Outcomes (2508.11800) | 2025-08 | 2,3,4,7 | exists, mechanism accurate | no | The paper's mechanism needs three things this project does not have: a model that outputs a probability (verbalized p_hat), a proper-scoring-rule reward r(p_hat… |
| 11 | Held-out counterfactual monitor from teacher-forced verdict probabilities: expected flip rate, boundary mass,… | Counterfactual Fairness in Text Classification through Robustness (1809.10610) | 2018-09 (a… | 2,4,6,7 | exists, partly accurate | maybe | It is an instrument, not a training change, so it cannot move unstable sets at matched hire rate by itself; and two of its load-bearing statistics are mis-speci… |
| 12 | Implicit-marker, salience-tiered counterfactual augmentation of the injection pipeline with a competence-prese… [gap] | Small Changes, Big Impact: Demographic Bias in LLM-Based Hiring Through Subtle Sociocultur… (2603.05189) | 2026-03 (v… | 2,3,5 | exists, partly accurate | maybe | The scarce resource is not prompts but model uncertainty, and this method supplies prompts. The 96% all-agree share arose on prompts the policy had NEVER seen (… |
| 13 | Nemotron reward profiling + Gaussian pass-rate curriculum with re-profiling at plateau | Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Age… (2512.20848) | 2025-12 (N… | 2,3,5 | exists, partly accurate | maybe | The method is mostly already in place, and the part that is new points the wrong way for a binary verdict. "Profile every prompt with the SFT checkpoint, drop p… |
| 14 | Paired, candidate-clustered CI and minimum-detectable-effect (power) analysis for the unstable-set share | Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations (2411.00640) | 2024-11 | 6,7 | exists, partly accurate | maybe | The proposal's premise is wrong for this repo: the paired, candidate-clustered interval it asks for is already what every decision CI is. `/ritom/scratch/cscs/z… |
| 15 | Permutation-Aware GRPO (PA-GRPO): one shared advantage baseline across all counterfactual variants of an item… | Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO (2603.21016) | 2026-03 (v… | 2,4,5 | exists, mechanism accurate | maybe | The consistency-aware reward as PA-GRPO specifies it (match the set's MODAL verdict, +1/-1, lambda=1) is label-free and, against this project's 0/+1 anchor rewa… |
| 16 | Post-hoc affine concept editing / LEACE-style linear erasure of the attribute direction at the decision positi… [gap] | Robustly Improving LLM Fairness in Realistic Settings via Interpretability ; LEACE: Perfec… (2506.10922) | 2025-06 ;… | 4,5,6,7,other | exists, mechanism accurate | maybe | A single global difference-in-means direction per attribute can only remove the population-mean shift that the attribute induces in the residual stream; it does… |
| 17 | Pre-registered checkpoint selection on a disjoint counterfactual dev split (never on the 450-pair benchmark) | A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibilit… (2504.07086) | 2025-04 (v… | 6,7 | exists, partly accurate | maybe | Observation 7 was not caused by a missing selection rule, so this rule cannot fix it. The final adapter (step 571) WAS the pre-registered checkpoint (EVAL_STAND… |
| 18 | Pre-training verifier audit: anchor teacher vs the guardrail judge (confusion matrix, error structure, FPR/FNR… [gap] | Checking the Checkers: A Synthesis of Verifier Noise and Robustness in RLVR ; Delay, Plate… (2610.17824) | 2026-10 ;… | 3,4,6,7 | exists, partly accurate | maybe | It is already done and already acted on, and when I re-ran its one open item it did not point at the anchor. reports/teacher_anchor/Qwen3.5-122B-A10B-GPTQ-Int4-… |
| 19 | Privileged-context on-policy self-distillation: teacher = same model on the attribute-free CV, student = same… [gap] | Qwen3.5-Omni Technical Report (Sec. 4.1 Thinker post-training, stage 2 On-Policy Distillat… (2604.15804) | 2026-04 (O… | 2,3,4,5,6,8 | exists, partly accurate | maybe | The headline metric rewards determinism, and the method as proposed removes the mechanism that produced it. An unstable set is any pair whose verdict differs ac… |
| 20 | Relabel the anchors with a judge aligned to the guardrail judge (GPT-4o directly, or the same teacher under th… [gap] | Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers ; Verif… (2510.00915) | 2025-10 (v… | 3,4,5,7 | exists, partly accurate | maybe | The symptom the relabel targets has not been shown to be the anchor's boundary. SFT-v3 already sits at its teacher's operating point on the benchmark: where GPT… |
| 21 | Replicate instability floor: re-run the attribute-free prompt under the benchmark's own decoding and read ever… [gap] | Instability Floors: Separating Bias from Noise in Fairness Audits of Clinical LLM Agents w… (2609.03221) | 2026-09 (v… | 2,6,7,other | exists, partly accurate | maybe | The method's premise does not hold for this benchmark's headline metric. The proposal reads "each pair decided many times" as stochastic replicate draws, but th… |
| 22 | Reward Expectation Matching (REM): paired-arm GRPO that matches the policy's expected reward across counterfac… | Aligning Language Models Towards Counterfactual Invariance (2610.22942) | 2026-10 | 4,5 | exists, mechanism accurate | maybe | In this setting REM collapses into the consistency reward the user already proposed, and what remains distinctive is either inert or harmful. (1) The reward is… |
| 23 | Reward Expectation Matching across counterfactual arms (Aligning LMs Towards Counterfactual Invariance) | Aligning Language Models Towards Counterfactual Invariance (2610.22942) | 2026-10 | 3,4,5,6 | exists, mechanism accurate | maybe | REM's per-response weight is alpha*(mu_own - mu_other)*(h_i - mu_own_LOO). With a 0/1 verdict and 16 draws, (h_i - mu_own_LOO) is identically zero for every gro… |
| 24 | Selection-aware reporting for checkpoint x arm choice: frozen shortlist, repeated split selection/confirmation… [gap] | Towards Reliable LLM Evaluation: Correcting the Winner's Curse in Adaptive Benchmarking (2605.05973) | 2026-05-07 | 2,3,7,other | unverified | unverified | — |
| 25 | Single-pass decision score: read p_norm(hire) at the forced decision position and sweep the threshold (ROC-sty… [gap] | Single-Pass Decision Scores Enable Rapid Classification with Adjustable Thresholds in Medi… (2610.23988) | 2026-10 ;… | 2,3,4,6,7 | exists, mechanism accurate | no | Its core is already this project's standard, not a new method. op-margin-v2 (docs/METRICS.md "Set stability at matched hire rate"; scripts/operating_point_contr… |
| 26 | Tail checkpoint averaging (LAWA / uniform soup of the last k adapters) as the evaluated artifact | When, Where and Why to Average Weights? (2502.06761) | 2025-02 (v… | 7 | exists, partly accurate | maybe | The thing that sank GRPO-v3-all is a monotone drift toward reject (25 of SFT-v3's 107 attribute-free hires became rejects; TPR guardrail -10.11 pp), not zero-me… |
| 27 | Thinking Machines on-policy distillation recipe: per-token advantage = -(sampled-token reverse KL) to the attr… | On-Policy Distillation | 2025-10 | 2,4,5 | exists, partly accurate | maybe | In this setting the recipe collapses into one of two things already on the table, and its headline promise (a usable gradient in the 96% all-agree groups) does… |
| 28 | AceReason/AceMath-RL strict on-policy single-update rule, hard-set filtering, and entropy-targeted temperature | AceReason-Nemotron: Advancing Math and Code Reasoning through Reinforcement Learning (1.0)… (2505.16400) | 2025-05 (1… | 2,3,7 | exists, partly accurate | no | The method's core, strict on-policy with one update per rollout, is the status quo: GRPO-v3-all (job 5116822) already trained exactly one optimizer step per rol… |
| 29 | Arity-matched replicate null + within-set permutation test + BCa ratio (AgentFairBench correction) [gap] | AgentFairBench: Do LLM Agents Discriminate When They Act? (2606.16723) | 2026-06 (v… | 2,6,7,other | exists, mechanism accurate | no | It is evaluation machinery that measures a different estimand from the project's headline metric, and it is uninterpretable on exactly the arms that matter. The… |
| 30 | Attribute-contrast at the decision token: contrastive-input / polarity-prompt contrastive decoding (CID, Promp… [gap] | PromptCD: Test-Time Behavior Enhancement via Polarity-Prompt Contrastive Decoding ; Surfac… (2602.20696) | 2026-02 ;… | 3,4,5,6 | exists, mechanism accurate | maybe | The "low-variance consistency signal" is arithmetically inert under the project's configuration. Both grpo.py and dapo.py leave scale_rewards at TRL's default '… |
| 31 | Bias-augmented consistency training (BCT) and Activation Consistency Training (ACT): self-targets from the mod… | Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought ; Consist… (2403.05518) | 2024-03 (v… | 3,4,5 | exists, partly accurate | maybe | The benchmark scores greedy verdicts, and SFT-v3's residual of 50 unstable sets (1.85% of 2,700) sits on borderline CVs where the hire/reject margin is tiny (ju… |
| 32 | Calibrate the anchor's 8-vote score to the guardrail judge: threshold or soft map chosen on a labelled calibra… [gap] | How to Correctly Report LLM-as-a-Judge Evaluations (2511.21140) | 2025-11-26… | 3,4,5,6 | unverified | unverified | — |
| 33 | Consistency Training: Bias-augmented (BCT, output-level SFT onto own clean-prompt response) / Activation Consi… [gap] | Consistency Training Helps Stop Sycophancy and Jailbreaks (BCT + ACT) ; Bias-Augmented Con… (2510.27062) | 2025-10 (B… | 2,3,4,5,7 | exists, partly accurate | maybe | The cheap (~1 GPU-h) offline BCT variant is a frozen-target distillation of SFT-v3 into itself, and the thing that makes consistency training work in the DeepMi… |
| 34 | Constrained GRPO: Lagrangian guardrails with scalarized STANDARDIZED advantages (TPR / undecided-rate as const… | Constrained Group Relative Policy Optimization (2602.05863) | 2026-02 (v… | 4,8 | exists, mechanism accurate | no | With a binary primary reward r_i = 1[decision_i == anchor] and the proposed cost c_TPR,i = 1[anchor == hire AND decision_i == reject], the cost is a determinist… |
| 35 | Counterfactual-set invariance reward inside GRPO (Reward Expectation Matching / Transformed REM on matched att… [gap] | Aligning Language Models Towards Counterfactual Invariance (2610.22942) | 2026-10 | 4,5,6 | unverified | unverified | — |
| 36 | Cross-counterfactual feedback-similarity reward (PS-GRPO group-similarity reward applied to the feedback field… [gap] | Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards ; Shapi… (2510.04392) | 2025-10 ;… | 2,4,5,8 | exists, partly accurate | no | Measured on the SFT-v3 audit parquet (outputs/raw/Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional.parquet, 450 pairs, 69 attributed rows per pair, lexic… |
| 37 | Cross-view self-reward with an EMA teacher: attributed draws rewarded against the attribute-free view's majori… [gap] | Co-rewarding: Stable Self-supervised RL for Eliciting Reasoning in Large Language Models (2508.00410) | 2025-08 (v… | 2,4,5 | exists, mechanism accurate | maybe | The Co-rewarding-specific machinery (cross-view pseudo-labels from 8 fresh draws, an EMA teacher, a vote-confidence gate) solves a problem this project does not… |
| 38 | EIS-GRPO: split one GRPO group across equivalent prompt variants and sum a global and a local advantage | J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization (2505.13346) | 2025-05 (v… | 4,5 | exists, partly accurate | no | EIS-GRPO's only novel signal is a non-zero advantage for a subgroup that is unanimous while a sibling variant of the same pair disagrees; everything else it doe… |
| 39 | Epistemic Independence Training (EIT): bias-cue-aware reward with balanced conflict (penalize following the at… [gap] | Making Bias Non-Predictive: Training Robust LLM Reasoning via Reinforcement Learning (2602.01528) | 2026-02 (v… | 4,5 | exists, mechanism accurate | no | In a binary verdict task EIT's three-way R_ind degenerates and is cancelled by GRPO's group normalization. Each group is one prompt, so it is entirely adversari… |
| 40 | Exact-logit difficulty scoring (PCL / GPS value model replaced by the policy's own decision-token probability) | Prompt Curriculum Learning for Efficient LLM Post-Training (2510.01135) | 2025-10 (a… | 2,3 | exists, mechanism accurate | maybe | As framed (train on prompts whose p_theta(anchor) is closest to 0.5) it scores the wrong difficulty for this objective and buys speed the project does not need.… |
| 41 | GDPO decoupled multi-reward normalization (only meaningful once a second NON-constant reward exists) | GDPO: Group reward-Decoupled Normalization Policy Optimization for Multi-reward RL Optimiz… (2601.05242) | 2026-01 | 4,5,8 | exists, partly accurate | no | In this project every candidate reward is a deterministic function of ONE binary decision token, so GDPO's mechanism has nothing to act on. With the current pai… |
| 42 | Grounded anchors plus a strict admission gate for synthesized pairs; never relax the gate to grow the pool (Su… [gap] | Survive or Collapse: The Asymmetric Roles of Data Gating and Reward Grounding in Self-Play… (2605.22217) | 2026-05 | 2,3,4,7 | exists, mechanism accurate | no | The paper's gate removes ill-posed SELF-GENERATED tasks in a proposer-solver loop whose failure mode is collapse; this project has no proposer, every anchor is… |
| 43 | Guardrails as one-sided non-inferiority bounds (lower CI bound inside the margin, intersection-union across gu… [gap] | Equivalence testing and run-count design for stochastic machine-learning experiments (2610.23700) | 2026-10 | 4,7,other | exists, partly accurate | maybe | It is evaluation-only and cannot move a single verdict: the drift toward reject that produced -10.11 pp on the TPR guardrail (observation 4) is a property of th… |
| 44 | HEIMAT: set-level Jensen-Shannon divergence across demographic-variant prompts at the prediction position (no… [gap] | A Heuristic Perspective on Debiasing Language Models (2608.00622) | 2026-08 | 2,4,5,6 | exists, mechanism accurate | maybe | HEIMAT's objective contains only an agreement term: it is minimized equally by a correct, a wrong, or a coin-flip shared verdict, so nothing in the method oppos… |
| 45 | Instance-level routing (I-SDPO / RLSD): zero-advantage (all-agree) groups train on the privileged consistency… | I-SDPO: Instance-Level Adaptive Self-Distillation Policy Optimization (2608.12957) | 2026-08 | 2,3,5 | exists, partly accurate | no | Under GRPO an all-agree group already contributes exactly zero gradient (advantage 0; only its tokens sit in the dapo denominator), so re-routing it cannot remo… |
| 46 | Interpolation toward SFT-v3 (LITI / WiSE-FT / rewarded soups) as a continuous hire-rate dial for matched-rate… | Rewarded soups: towards Pareto-optimal alignment by interpolating weights fine-tuned on di… (2306.04488) | 2023-06 | 4,6,7 | exists, partly accurate | maybe | The project's decision metric already IS a matched-hire-rate comparison: op-margin-v2 (scripts/operating_point_control.py, docs/EVAL_STANDARD.md) moves SFT-v3 t… |
| 47 | Intervention-consistency flip-rate monitor on held-out counterfactual sets with a control-pair noise floor and… | When Names Change Verdicts: Intervention Consistency Reveals Systematic Bias in LLM Decisi… (2603.18530) | 2026-03 (I… | 6,7 | exists, partly accurate | maybe | It is a measurement, not a training change, and as a measurement it is a lower-powered copy of the benchmark on a different population. With ~128-440 held-out p… |
| 48 | J1 verdict-consistency reward: reward only when all orderings/variants of the same item are judged correctly | J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning (2505.10320) | 2025-05 (v… | 4,5 | exists, mechanism accurate | no | The J1 "AND" reward does not change what the policy learns, only how prompts are weighted. The attributed completion a_i and its control partner c_i are indepen… |
| 49 | Learnability-based online prompt selection under the CURRENT policy (Learning-Zone Energy; HIVE; GPS; Prompt R… | Learning-Zone Energy: Online Data Selection for Efficient RL Post-Training ; Train at Movi… (2605.17003) | 2026-05 (v… | 2,3 | exists, partly accurate | no | The project has already run this method's premise and it came up empty. DAPO-v3-all (job 5119438, 2026-10-22, outputs/dapo/qwen3.5-9b_en_only_dapo_v3_all/dynami… |
| 50 | Learnability-rewarded proposer applied to the owned job-CV generator (AZR / R-Zero / SPICE reward family; PROP… [gap] | Absolute Zero: Reinforced Self-play Reasoning with Zero Data ; R-Zero: Self-Evolving Reaso… (2505.03335) | 2025-05 | 2,3 | exists, mechanism accurate | maybe | The method cures a scarcity that is not the binding constraint. GRPO-v3-all did not fail for lack of split groups; it failed the TPR guardrail (-10.11 pp) and o… |
| 51 | Learning from zero-variance groups via entropy/confidence-shaped advantages (RL-ZVP, LENS, NSR, NGRPO) | No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via… (2509.21880) | 2025-09 (v… | 2,4,5 | exists, partly accurate | no | The method's lever is per-token entropy shaping, and that lever is absent here. In a 16/16 all-agree group the decision token has p(sampled verdict) close to 1… |
| 52 | Learning-rate cooldown (warmup-stable-decay or cosine-to-min) with a pre-registered final-checkpoint rule and… | A Sober Look at Progress in Language Model Reasoning: Pitfalls and Paths to Reproducibilit… (2504.07086) | 2025-04 (v… | 7 | exists, partly accurate | maybe | Observation 7 is not evidence of an optimizer random walk; it is evidence of a binarized decision rule with roughly one-CV granularity sitting on its thresholds… |
| 53 | Masked-counterfactual logit fusion at decode time (COFT); lambda=1 is the attribute-free (redaction) decoding… [gap] | COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Langu… (2605.30641) | 2026-05 | 4,5,6 | exists, partly accurate | no | Everything COFT would contribute here is either already free or already in the plan. (a) The lambda=1 "achievable floor" is not a method: with all attribute val… |
| 54 | Multi-environment simultaneous RL with environment-specific rewards (Nemotron 3 recipe) | Nemotron 3 Nano: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Age… (2512.20848) | 2025-12 (N… | 4,5,6 | exists, partly accurate | no | The Nemotron evidence is about forgetting OTHER domains when one domain is trained; the TPR failure here is not forgetting. GRPO-v3-all's in-run anchor accuracy… |
| 55 | Nemotron-Cascade epoch-level dynamic filtering with stochastic re-admission (plus strict on-policy, no-KL GRPO… | Nemotron-Cascade: Scaling Cascaded Reinforcement Learning for General-Purpose Reasoning Mo… (2512.13607) | 2025-12 | 2,3,5,7 | exists, mechanism accurate | no | Epoch-level filtering has nothing to act on in a one-epoch run: every one of the 9,140 prompts is drawn once, so dropping a prompt after its chunk changes nothi… |
| 56 | OPCD-style frozen context teacher: teacher = frozen SFT-v3 (adapter disabled) on the attribute-free prompt, st… | On-Policy Context Distillation for Language Models (2602.12275) | 2026-02 (v… | 2,4,5,7 | exists, mechanism accurate | maybe | The fixed point is SFT-v3's own soft verdict distribution on the clean CV, and the objective is non-sharpening, so it attacks the wrong part of the metric. Unde… |
| 57 | On-policy prompt replay / experience buffer of high-signal prompts (Prompt Replay, ExGRPO, DOTS rollout replay… | Prompt Replay: Speeding up GRPO with On-Policy Reuse of High-Signal Prompts (primary); ExG… (2603.21177) | 2026-03 (P… | 2,3 | exists, partly accurate | maybe | Prompt Replay's premise is that prompts whose pass rate sits near 0.5 carry the most learnable signal, which holds in math where the label is verifiable. Here t… |
| 58 | Output-level counterfactual consistency / privileged-context self-distillation: train on the attributed prompt… [gap] | Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought ; Consist… (2403.05518) | 2024-03 (v… | 2,3,4,5,7 | exists, partly accurate | maybe | SFT-v3 already IS this method with a teacher label. Its 62,761 training rows are (attributed prompt, 12 variants per pair) -> an invariant response carrying the… |
| 59 | PAPO decoupled advantage normalization (outcome advantage over all rollouts; secondary advantage normalized on… [gap] | PAPO: Stabilizing Rubric Integration Training via Decoupled Advantage Normalization (2603.26535) | 2026-03 (v… | 2,8 | exists, partly accurate | no | There is nothing to decouple. The project's reward list is exactly [decision_reward = 1.0, format_reward = 0.2] (src/hiring_bias_mitigation/mitigation/grpo.py:6… |
| 60 | Paraphrase-null baseline for counterfactual flip rates and per-sample metrics ('Compared to What?') [gap] | Compared to What? Baselines and Metrics for Counterfactual Prompting ; Counterfactual Fair… (2605.01048) | 2026-05 (v… | 6,7 | exists, partly accurate | no | The project's decision statistic is already a difference against a reference arm: the trained checkpoint's unstable-set share vs SFT-v3's at MATCHED hire rate.… |
| 61 | Pooled pair groups: attribute-free and attributed completions of one pair in ONE advantage group | Information-Consistent Language Model Recommendations through Group Relative Policy Optimi… (2512.12858) | 2025-12 (v… | 2,3,4,5 | exists, partly accurate | no | The signal this method uniquely adds barely exists here, and its reward on the neutral arm strengthens the exact mechanism that failed the TPR guardrail. Joinin… |
| 62 | Pre-registration diagnostics for privileged-information effectiveness: sampled-token reverse-KL D and error-st… | Predicting Privileged Information Effectiveness for On-Policy Self-Distillation (2610.22512) | 2026-10 | 3,5,6,7 | exists, mechanism accurate | maybe | As specified, the four checks are circular or degenerate for this task, so the scoring pass has no veto power over anything in the plan. (i) "Tipped" is already… |
| 63 | Precision fixes for the LoRA update: FP16 or FP32 for the served/merged policy, FP32 lm_head, and reading TRL'… | Defeating the Training-Inference Mismatch via FP16 (primary); also: The Art of Scaling Rei… (2510.26788) | 2025-10 (P… | 7,other | exists, partly accurate | no | The project's own instrumentation already refutes the mechanism for this stack. TRL's mismatch metrics are in the GRPO-v3-all log (job 5116822, /ritom/scratch/c… |
| 64 | Privileged-information self-distillation routed to all-wrong groups (I-SDPO / OPSD / SRPO / TTPO family) | I-SDPO: Instance-Level Adaptive Self-Distillation Policy Optimization (2608.12957) | 2026-08 | 2,3,4,5 | exists, partly accurate | maybe | The routed population is small and is the one place the anchor is least trustworthy. From the GRPO-v3-all log (job 5116822) the all-wrong share of groups is bou… |
| 65 | SDPO hybrid: convex blend of the anchor-reward policy loss and self-distillation from a privileged self-teache… | Reinforcement Learning via Self-Distillation (2601.20802) | 2026-01 (a… | 2,4,5 | exists, partly accurate | no | Dense in tokens is not dense in information. SDPO's signal comes from a teacher with PRIVILEGED context (the solution or feedback), which makes the teacher's di… |
| 66 | Sequential probe-then-extend rollout allocation (SARA early abandonment + Reinforce-Ada balanced extension, SP… | Reinforce-Ada: An Adaptive Sampling Framework under Non-linear RL Objectives (2510.04996) | 2025-10 (v… | 2 | exists, partly accurate | no | The benchmark metric is defined on GREEDY decisions (docs/EVAL_STANDARD.md: temperature 0.0, "unstable sets, greedy decisions"), but the proposal's entire gain… |
| 67 | Signed per-set estimand: regression on per-pair P(hire) differences with clustered SE (and McNemar on discorda… [gap] | Compared to What? Baselines and Metrics for Counterfactual Prompting ; Instability Floors:… (2605.01048) | 2026-05 (v… | 4,5,6,7,other | exists, partly accurate | maybe | It is a change of estimand, not a fix. Nothing in it touches the policy, the reward, or the prompt set, so it cannot address observations 4 and 5 (the reject dr… |
| 68 | Soft / calibrated anchor targets: Brier-type reward against the teacher vote fraction (RLCR) instead of the bi… | Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (2507.16806) | 2025-07 | 4,5,7 | exists, partly accurate | maybe | Observation 4 is a systematic offset of the anchor, not vote noise, so a soft target aims at the wrong thing. The same teacher anchor prompt rejects 79.1% (sing… |
| 69 | Strong-to-weak on-policy distillation from a larger open Qwen3.5 teacher given the attribute-free CV (the Qwen… [gap] | Qwen3 Technical Report (Sec. 4.5 Strong-to-Weak Distillation; Sec. 4.7 Table 21) ; Llama-N… (2505.09388) | 2025-05 (Q… | 2,4,7,other | exists, partly accurate | maybe | The headline metric rewards determinism across attribute values, but the optimum of a per-token KL-to-teacher objective is the teacher's own soft decision distr… |
| 70 | Targeted label acquisition where the policy confidently contradicts the anchor (active relabeling / appeals on… [gap] | When Self-Belief Misleads: Active Label Acquisition for Reinforcement Learning with Verifi… (2605.25864) | 2026-05 (v… | 2,3,4,5 | exists, partly accurate | maybe | The method is built for scarce gold labels and scattered label errors, and this project has neither. The anchor is already an 8-draw majority of Qwen3.5-122B (8… |
| 71 | Two training seeds per arm plus within-run checkpoint spread as a noise floor | DataFlex-RL: An Evaluation Platform for RLVR Data Policies (2609.06107) | 2026-09 | 1,7 | exists, partly accurate | maybe | Seed averaging targets the wrong noise source. The 525-vs-571 flip is not between-run noise: the two checkpoints share 444/450 attribute-free verdicts and the c… |
| 72 | Affine concept editing / activation steering on attribute directions (inference-time, label-free) | Robustly Improving LLM Fairness in Realistic Settings via Interpretability (2506.10922) | 2025-06 | 4,6,other | exists, partly accurate | maybe | The paper's success metric is the group-mean interview-rate gap along ONE linear direction for race/gender names on untuned chat models; this project's headline… |
| 73 | Attribute-redaction decoding baseline and its leakage ceiling (de-identified decoding) [gap] | Beyond the Name: Demographic Leakage in De-Identified Résumés and Evaluation Artifacts in… (2609.16501) | 2026-09 ;… | 5,6,other | exists, partly accurate | no | The benchmark decodes greedily (docs/EVAL_STANDARD.md:67), so decoding every variant of a within-group set from the one shared attr_free prompt gives identical… |
| 74 | Balanced band filtering plus retirement of mastered prompts (Online Difficulty Filtering; ScaleRL zero-varianc… | Online Difficulty Filtering for Reasoning Oriented Reinforcement Learning (ODF) / The Art… (2504.03380) | ODF: 2025-… | 2,3 | exists, partly accurate | no | The symmetric band discards exactly the rows this project built its data to find, on the side that already limits training. Under SFT-v3's own sampled error rat… |
| 75 | Bias-aware GRPO data design (JudgeBiasBench): verified bias-injected variants with unchanged labels inside the… | Toward Robust LLM-Based Judges: Taxonomic Bias Evaluation and Debiasing Optimization (2603.08091) | 2026-03 (v… | 5,8 | exists, partly accurate | maybe | The paper's actual mechanism (train on bias-injected variants whose label is the un-injected verdict) is already 100% of this project's GRPO prompt set: every a… |
| 76 | Consensus-gated anchors from a judge panel with calibrated weighted voting; abstain (zero advantage) where jud… [gap] | Can We Trust LLM Judges: A Study of Capability-Dependent Biases and Multi-Judge Ensemble f… (2609.12002) | 2026-09 ;… | 2,4,5 | exists, mechanism accurate | maybe | The headline metric is within-pair invariance, not accuracy, and a coin-flip anchor is not noise for invariance: every variant of a pair shares one fixed target… |
| 77 | DLER recipe: batch-wise reward normalization and update-selective merging | DLER: Doing Length pEnalty Right - Incentivizing More Intelligence per Token via Reinforce… (2510.15110) | 2025-10 | 2,7,8 | exists, mechanism accurate | no | Of DLER's four training parts, three are inert or already taken here: dynamic sampling is DAPO-v3-all's core change (docs/DAPO_V3.md), clip-higher is inert with… |
| 78 | Debiasing-DPO: counterfactual preference pairs (own neutral-context response preferred over own biased-context… | Mitigating LLM biases toward spurious social contexts using direct preference optimization (2604.02585) | 2026-04 (v… | 4,5,7 | exists, partly accurate | maybe | The method is not new here: the repo's DPO recipe already IS "counterfactual pairs + SFT anchor" (every DPO config uses loss_type [sigmoid, sft] with weights 1.… |
| 79 | Difficulty sharing across related prompts (graph-structured online difficulty estimation; LEEPS latent neighbo… | Efficient RLVR Scheduling via Graph-Structured Online Difficulty Estimation (bundled with:… (2608.17941) | 2026-08; L… | 2,3 | exists, mechanism accurate | no | The method solves a budget problem this project does not have, and its "success" points the wrong way. Sharing difficulty across siblings pays off only when per… |
| 80 | Difficulty-adaptive rewriting of CVs toward the verdict boundary with re-anchoring (Synthetic Data RL curricul… [gap] | Synthetic Data RL: Task Definition Is All You Need (2505.17063) | 2025-05-18 | 2,3 | unverified | unverified | — |
| 81 | EMA policy weights: evaluate the EMA, and optionally use it as the KL anchor (WARP / EMA-PG) | WARP: On the Benefits of Weight Averaged Rewarded Policies (2406.16768) | 2024-06 | 4,7 | exists, partly accurate | maybe | Observation 7 is a detection-limit problem, not a weight-trajectory-noise problem, and EMA only treats the latter. Steps 525 and 571 already agree on 444/450 at… |
| 82 | Faithful GRPO: constraint on a secondary signal via Lagrangian dual ascent with per-signal group normalization [gap] | Faithful GRPO: Improving Visual Spatial Reasoning in Multimodal Language Models via Constr… (2604.08476) | 2026-04 | 4,8 | exists, partly accurate | no | The method's premise is a secondary signal that varies independently of the task signal within a rollout group. With a 1-bit verdict per completion that dimensi… |
| 83 | Flip-Flop Consistency (F2C): consensus cross-entropy over prompt variants + JSD among confident variants + asy… [gap] | Flip-Flop Consistency: Unsupervised Training for Robustness to Prompt Perturbations in LLM… (2510.14242) | 2025-10 (v… | 2,4,5,6 | exists, mechanism accurate | maybe | The paper's own ablation (Table: Qwen2.5-3B) shows CCE-only gives +11.68% agreement / +8.36% F1 versus +11.62% / +8.94% for full F2C, and the soft-alignment bas… |
| 84 | GPG effective-sample rescaling: multiply the loss by B/(B-M) where M completions sit in all-agree groups | GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning (2504.02546) | 2025-04 | 2,7 | exists, partly accurate | no | The premise that zero-advantage groups diluted the gradient ~25x is false for the gradient that reached the optimizer. Parsing all 571 steps of GRPO-v3-all's lo… |
| 85 | GRESO probabilistic pre-rollout skipping from zero-variance history | Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Ro… (2506.02177) | 2025-06 | 2,3 | exists, partly accurate | no | GRESO is a rollout-cost saver, not a signal creator: its own paper reports accuracy parity with post-rollout dynamic sampling (within +-0.5 pt across six benchm… |
| 86 | GSPO: sequence-level, length-normalised importance ratio and clipping (the Qwen lineage's named RL optimiser f… [gap] | Group Sequence Policy Optimization (Zheng, Liu, Li, Chen, Yu, Gao, Dang, Liu, Men, Yang, Z… (2507.18071) | 2025-07 | 7,other | exists, mechanism accurate | no | GSPO is mathematically identical to the current recipe and probably degenerate in the planned one. With one optimizer step per rollout, pi_theta = pi_old at the… |
| 87 | History-aware baseline: Discounted Beta-Bernoulli (DBB) advantage seeded with SFT-v3's sampled error rate | Discounted Beta-Bernoulli Reward Estimation for Sample-Efficient Reinforcement Learning wi… (2603.18444) | 2026-03 (v… | 2,3 | exists, partly accurate | no | In this recipe every prompt is visited once (one epoch, 572 steps), so the Beta posterior is never updated and lambda never discounts anything: DBB degenerates… |
| 88 | In-training representation-invariance penalty on the decision-position hidden state: adversarial attribute hea… [gap] | Towards bias-free recruitment: Adversarial contrastive tuning for LLM-based resume screeni… (doi:10.1016/j.infsof.2026.108040) | 2026-01 (o… | 2,4,5 | exists, partly accurate | no | The penalty constrains the wrong object at the wrong granularity. (a) Granularity: GRL/MMD enforce DISTRIBUTIONAL independence of the decision-position state fr… |
| 89 | J1 verdict-consistency reward: pair the rollouts of both counterfactual orderings in the same batch and pay +1… [gap] | J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning (2505.10320) | 2025-05 (v… | 4,5 | exists, partly accurate | maybe | The J1-specific mechanism adds no information that the project's already-written plan does not have, and the form named in the title cannot touch observation 4.… |
| 90 | Knapsack RL: per-prompt rollout budget allocation by learnability value | Knapsack RL: Unlocking Exploration of LLMs via Optimizing Budget Allocation (2509.25849) | 2025-09 | 2,3 | exists, mechanism accurate | no | The mechanism Knapsack RL pays for (many rollouts on a near-unanimous prompt so that a rare, differently-reasoned success appears) is inert when the completion… |
| 91 | Lagrangian operating-point control: Constrained GRPO with standardized-advantage scalarization (hire-rate / TP… | Constrained Group Relative Policy Optimization (2602.05863) | 2026-02 (v… | 4,6,7,8 | exists, partly accurate | no | The constraint it would enforce was never violated, on either candidate measurement, during the very run whose TPR guardrail failed, so the multiplier would hav… |
| 92 | Magnitude-matched paraphrase / neutral-edit control arm: compare attribute-induced change against meaning-pres… [gap] | Compared to What? Baselines and Metrics for Counterfactual Prompting (2605.01048) | 2026-05 (v… | 4,5,6,other | exists, mechanism accurate | maybe | The project's headline metric is already a difference between two arms (treated model vs SFT-v3) at matched hire rate, so generic CV-edit sensitivity is common… |
| 93 | Noise-corrected GRPO (asymmetric Bernoulli flip correction) and the 'imperfect verifier' tolerance evidence | Noise-corrected GRPO: From Noisy Rewards to Unbiased Gradients ; An Imperfect Verifier is… (2510.18924) | 2025-10 (v… | 4,7,other | exists, partly accurate | no | The noise model does not match this project's noise. NC-GRPO assumes each completion's reward is flipped independently (xi independent of (q,o); synthetic runs… |
| 94 | Null-reward control run (shuffled anchors) to separate reward-driven change from reward-independent sharpening | Spurious Rewards: Rethinking Training Signals in RLVR (2506.10947) | 2025-06 | 4,7,8 | exists, partly accurate | no | The headline metric is immune by construction to the effect this control is meant to detect. The audit is greedy (temperature 0.0 in both audit configs) and op-… |
| 95 | OPSD (Self-Distilled Reasoner): privileged-information self-teacher held at the INITIAL policy, forward KL ove… | Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models (2601.18734) | 2026-01 (v… | 2,4,5,7 | exists, partly accurate | maybe | The benchmark decides every prompt by GREEDY decoding on merged weights (docs/EVAL_STANDARD.md line 67, docs/METRICS.md "Within-group unstable sets ... greedy d… |
| 96 | Online category-level curriculum bandit (Self-Evolving Curriculum; AdaRFT target difficulty) | Self-Evolving Curriculum for LLM Reasoning (SEC); paired entry: Efficient Reinforcement Fi… (2505.14970) | SEC 2025-0… | 2,3 | exists, partly accurate | no | With a binary reward, SEC's arm value (mean \|normalised advantage\|) is just the group split rate, i.e. how often the current policy disagrees with the ANCHOR.… |
| 97 | Oscillation and signal-exhaustion diagnostics from GRPO dynamics models (damping ratio, advantage-collapse rat… | Predictable GRPO: A Closed-Form Model of Training Dynamics (2606.30789) | 2026-06 | 2,6,7 | exists, partly accurate | maybe | There is no oscillation in GRPO-v3-all's training data for a damped-oscillator model to fit, and the "oscillation" of obs 7 is evaluation granularity, not dynam… |
| 98 | Pairwise counterfactual-invariance objectives inside GRPO: Reward Expectation Matching across matched attribut… [gap] | Aligning Language Models Towards Counterfactual Invariance (2610.22942) | 2026-10 | 2,4,5 | exists, partly accurate | no | The survey's central "fit" claim is wrong for the paper's actual formulation. REM's per-response reward is r_i^RM - alpha*(rbar_{-i} - rbar')*(r_i - rbar_{-i}):… |
| 99 | Pre-registered KL budget to SFT-v3 as a checkpoint exclusion / early-stop rule | RL's Razor: Why Online Reinforcement Learning Forgets Less (2509.04259) | 2025-09 | 4,6,7 | exists, partly accurate | no | In GRPO-v3-all the KL to SFT-v3 did not move while the TPR did, and the two audited checkpoints order the wrong way. TRL's k3 KL (beta 0.04) sat flat at 0.016-0… |
| 100 | ProRL / ProRL v2: KL trust region with periodic reference and optimizer resets, REINFORCE++-baseline with glob… | ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Mod… (2505.24864) | 2025-05 (a… | 2,7 | exists, partly accurate | no | ProRL's one distinctive ingredient, a KL trust region with periodic reference/optimizer resets, has nothing to act on here and points the wrong way on the guard… |
| 101 | Resolution diagnostics for paired, clustered comparisons: required N* and resolution ratio q = N/N* for each a… [gap] | Resolution Diagnostics for Paired LLM Evaluation (2605.30315) | 2026-05 | 7,other | exists, mechanism accurate | maybe | It is a power calculation, not an intervention: nothing it outputs changes the trained policy, the unstable-set share, or any of the four guardrails. The team a… |
| 102 | Response Matching (JSD regularizer on next-token distributions under the counterfactual prompt) inside GRPO, a… [gap] | Aligning Language Models Towards Counterfactual Invariance (2610.22942) | 2026-10 | 2,4,5 | exists, mechanism accurate | maybe | Here the output prefix `{"decision": "` is a fixed string (format reward 0.200 with std 0.0006), so the decision-token distribution under the attributed and the… |
| 103 | Scale rollouts per prompt (BroRL N=512; IsoCompute; F-GRPO focal weighting) | BroRL: Scaling Reinforcement Learning via Broadened Exploration (2510.01180) | 2025-10 (B… | 2 | exists, mechanism accurate | no | BroRL's mechanism is coverage: in a huge output space, more rollouts find rare correct trajectories whose reinforcement moves mass off wrong ones. Here the rewa… |
| 104 | SelfCI complementary two-teacher product-of-experts: distil toward the geometric mean of an attribute-free (in… | It Takes Two: Complementary Self-Distillation for Contextual Integrity in LLMs (2605.20258) | 2026-05 | 2,4,5 | exists, mechanism accurate | maybe | The benchmark is greedy-decoded (configs/audit/*gh200.yaml temperature 0.0; docs/EVAL_STANDARD.md "greedy decisions"), so an unstable set is an argmax flip, and… |
| 105 | Top-entropy token masking (`top_entropy_quantile=0.2`) - entropy control that fits; entropy bonuses/Clip-Cov/K… | Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning… (2506.01939) | 2025-06 (v… | 2,7 | exists, partly accurate | no | The claimed mechanism inverts on this output format. Measured on GRPO-v3-all's own audit outputs (outputs/raw/Qwen3.5-9B--en--grpo--adapter--en_only_grpo_v3_all… |
| 106 | Variance-reduced OPD estimators: vOPD control-variate baseline (= per-token negative reverse KL from the same… | KL for a KL: On-Policy Distillation with Control Variate Baseline (2605.07865) | 2026-05 | 2,7 | exists, partly accurate | no | vOPD reduces the variance of an estimator this project does not use and, if it ever adopts a clean-prompt self-distillation loss, should not use in that form. T… |
| 107 | Weight-space smoothing of the policy: EMA or averaging of the last k LoRA deltas (WARP-style) before the audit | WARP: On the Benefits of Weight Averaged Rewarded Policies (2406.16768) | 2024-06 | 7 | exists, partly accurate | maybe | The oscillation this targets is mostly not in weight space, so averaging attacks the smallest of three noise sources. Steps 525 and 571 agree on 444/450 attribu… |
| 108 | Zero-variance-group advantage shaping without extra rollouts (RL-ZVP entropy-guided; AVSPO virtual samples; AC… | No Prompt Left Behind: Exploiting Zero-Variance Prompts in LLM Reinforcement Learning via… (2509.21880) | 2025-09 (v… | 2,3,6 | exists, partly accurate | unverified | — |
| 109 | C2PO: reference-free causal-contrastive margin loss between an unbiased response and a shortcut-driven counter… [gap] | C2PO: Diagnosing and Disentangling Bias Shortcuts in LLMs (2512.23430) | 2025-12 | 4,5,8 | exists, mechanism accurate | no | On this task C2PO degenerates into a reference-free hard-label push on the verdict token that is a strictly less-regularized sibling of the DPO arm the project… |
| 110 | CROP-style counterfactual-sensitivity token selection: supervise only the completion positions whose probabili… | CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation (2608.13387) | 2026-08 (v… | 5,8 | exists, partly accurate | no | The loss CROP would re-weight does not exist in this pipeline. CROP allocates a per-token on-policy DISTILLATION signal; here (src/hiring_bias_mitigation/mitiga… |
| 111 | Calibration monitoring and optional calibration-aware reward (RLCR Brier term) on p(hire) | Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty (2507.16806) | 2025-07 | 4,6,7 | exists, partly accurate | maybe | Calibration and counterfactual stability pull in opposite directions on exactly the rows that decide the headline. A set is unstable when p_hire sits near 0.5 s… |
| 112 | Class-balanced dynamic sampling for binary-verdict GRPO (BD-GRPO from eXTC): per-class oversample -> drop zero… | Structured Prompt Optimization Meets Reinforcement Learning for Global and Local Interpret… (2605.29076) | 2026-05 (v… | 2,4 | exists, mechanism accurate | no | The mechanism BD-GRPO fixes is absent here, and the part of it that applies is already built. eXTC's gain came from an imbalanced dataset (MIMIC) where class-ag… |
| 113 | Consistency-labelled preference optimization (DPO/KTO with chosen = own attribute-free verdict), i.e. causal-c… | Reasoning Elicitation in Language Models via Counterfactual Feedback ; Sycophantic Agreeme… (2410.03767) | 2024-10 (v… | 4,5 | exists, partly accurate | maybe | The relabelling is a no-op on every pair this project's DPO pipeline already builds. A pair exists only on an attribute-tipped row, defined as a row where more… |
| 114 | Context-rich LLM-rewritten counterfactuals with uncertainty-based quality filtering (Context-CDA) [gap] | Context-Aware Counterfactual Data Augmentation for Gender Bias Mitigation in Language Mode… (2602.09590) | 2026-02 | 2,other | exists, partly accurate | no | The method's premise has no analog here. Context-CDA exists because template word-swaps damage a language-modeling objective (GPT-2 lost ~8 LMS points) and look… |
| 115 | Contrast-calibrated privileged teachers: isolate the attribute-specific part of the teacher-student discrepanc… | RLCSD: Reinforcement Learning with Contrastive On-Policy Self-Distillation (2606.11709) | 2026-06 (v… | 4,5 | exists, partly accurate | no | The contrast does not isolate what it claims, and the thing it would isolate is already a one-line mask here. Write the log-prob of a sampled token under the th… |
| 116 | Counterfactual ensemble decoding across all attribute values of a set (CED / CHOIR style) as an upper-bound or… [gap] | Mitigating Bias in Large Vision-Language Models via Counterfactual Ensemble Decoding ; CHO… (2608.21415) | 2026-08 ;… | 5,6,other | exists, partly accurate | no | The proposal splits into an oracle and a training method, and neither survives contact with this benchmark. (1) The oracle is a tautology, not an upper bound: d… |
| 117 | Expert-then-distill post-training (Qwen3-Coder-Next Sec. 4.2 and Qwen3.5-Omni stage 1): train RL specialists,… [gap] | Qwen3-Coder-Next Technical Report (Sec. 4.2 Expert Models: 4.2.3 single-turn QA expert, 4.… (2603.00729) | 2026-02 (C… | 4,7,other | exists, partly accurate | no | The deployed quantity here is one token. Distillation cannot give the student a decision distribution better than its teacher(s): a student distilled from the G… |
| 118 | Fairness Reward Model (FRM) / attribute-mention penalty on the feedback text as a secondary reward | Guiding LLM Decision-Making with Fairness Reward Models (2507.11344) | 2025-07-15 | 4,5 | unverified | unverified | — |
| 119 | Fairness Reward Model (FRM): a 1B process reward model scoring rationale text for reliance on protected attrib… [gap] | Guiding LLM Decision-Making with Fairness Reward Models (2507.11344) | 2025-07 | 8 | exists, mechanism accurate | no | The FRM scores rationale text, but in this project the bias lives in the verdict token and is invisible in the rationale. Verified on disk: summary.attribute_me… |
| 120 | GRPO-CARE gated consistency bonus: reference-model likelihood of the verdict given the rationale, compared to… [gap] | GRPO-CARE: Consistency-Aware Reinforcement Learning for Multimodal Reasoning (2506.16141) | 2025-06 | 8 | exists, mechanism accurate | no | The bonus has no fairness content and, in this output format, no reasoning content either: the policy emits "decision" BEFORE "feedback", so P_ref(decision \| f… |
| 121 | Generalized Knowledge Distillation / context distillation baselines (older): on-policy JSD distillation with a… | On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (2306.13649) | 2023-06 | 4,other | exists, partly accurate | no | The project has already run the closest form of baseline (a) and it was null: docs/RFT_CONTROL.md (job 5066776) did a second SFT stage on the student's own anch… |
| 122 | Importance-ratio shaping family (CISPO, GSPO, SAPO, VESPO, LUSPO, GMPO, INTELLECT-2 delta): inert at one step… | MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention (introduces CIS… (2506.13585) | 2025-06 (p… | 7,other | exists, partly accurate | no | The policy ratio pi_theta/pi_old is identically 1 in the current recipe, so every member of this family computes the same REINFORCE gradient. Verified in the in… |
| 123 | Inference-time affine concept editing of race/gender directions for hiring decisions (training-free; or as a t… [gap] | Robustly Improving LLM Fairness in Realistic Settings via Interpretability (2506.10922) | 2025-06-12 | 4,6,other | unverified | unverified | — |
| 124 | Label-free spectral erasure for implicit cues (H-SAL): project out the decision-position directions that co-va… [gap] | Debiasing Without Protected Attributes: Latent Concept Erasure from Textual Profiles ; [ba… (2606.12088) | 2026-06 | 5,other | exists, partly accurate | no | H-SAL's one selling point, erasure WITHOUT protected-attribute labels, buys nothing here: every one of the 9,140 attributed prompts has a known attribute, a kno… |
| 125 | LoRA-specific RL findings: rank is nearly irrelevant for RL, lr ~10x full fine-tuning, orthonormal init stabil… | LoRA Without Regret (blog); Geometry-Preserving Orthonormal Initialization for Low-Rank Ad… (2606.31813) | 2025-09 (b… | 7,other | exists, partly accurate | no | Nothing in it is an intervention on the project's actual bottlenecks. Every setting it endorses is already in place: `DEFAULT_LORA_TARGETS` covers all seven pro… |
| 126 | NeMo-RL ProRLv2 bundle: importance-sampling correction (seq-mask-tis / ICE-POP / TIS) and the Nemotron health… | NVIDIA-NeMo/RL GitHub releases v0.4.0 (2025-12-01), v0.5.0 (2026-01-31), v0.6.0 (2026-04-3… (2512.20848) | 2025-12 to… | 7,other | exists, partly accurate | no | It is already running and already measured to be immaterial. TRL 1.15 defaults to vllm_importance_sampling_correction=True with mode "sequence_mask" and C_max=3… |
| 127 | Noise-robust reward corrections for a ~80%-reliable anchor (asymmetric backward/forward correction; noise-corr… | Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers (2510.00915) | 2025-10 (v… | 4,7 | exists, partly accurate | no | With a two-valued reward (decision 1/0 plus a constant 0.2 format term), every per-sample function of the observed reward is affine in it, so backward correctio… |
| 128 | Nonlinear (kernel/HSIC) post-hoc concept erasure at the decision position, for the case where the attribute is… [gap] | Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure ; Nonlinear Concep… (2603.07529) | 2026-03 (v… | 5,other | exists, partly accurate | no | The erasure objective is the wrong quantity for this benchmark. HSIC(h', A) -> 0 (Obliviator) and class-conditional density matching (LEOPARD) enforce MARGINAL… |
| 129 | Off-benchmark checkpoint selection from training-reward uncertainty on the hardest prompts (UGCS), so the 450-… [gap] | Uncertainty-Guided Checkpoint Selection for Reinforcement Finetuning of Large Language Mod… (2511.09864) | 2025-11 | 2,7 | exists, partly accurate | no | UGCS ranks checkpoints by their training reward on the hardest training prompts, and here the training reward is "match the noisy anchor" (80% inter-judge agree… |
| 130 | On-policy distillation (reverse-KL) to consolidate RL specialists / checkpoints / seeds into one model (DeepSe… | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence ; Qwen3.5-Omni Te… (2606.19348) | 2026-04 (a… | 1,7 | exists, partly accurate | no | The quantity that drives every metric here is one binary token (the "hire"/"reject" decision) inside a 40-token JSON. For a binary variable, reverse KL to a mix… |
| 131 | Online Label Refinement: switch a pair's reward from the anchor to the policy's own stabilised attribute-free… [gap] | Can LLMs Learn to Reason Robustly under Noisy Supervision? (2604.03993) | 2026-04 | 2,4,5 | exists, partly accurate | no | OLR assumes the policy's majority vote recovers a TRUE label that the noisy supervision hides. Here there is no such label to recover: the anchor already IS a G… |
| 132 | Pass@k / pass^k group reward with analytical advantages | Pass@k Training for Adaptively Balancing Exploration and Exploitation of Large Reasoning M… (2508.10751) | 2025-08 | 2,other | exists, partly accurate | no | For a binary reward the method is not a new objective: with N=16 draws, every zero-mean advantage scheme fixes \|A_neg\|/\|A_pos\| = N_pos/N_neg inside a group,… |
| 133 | Probe-gated inference-time steering (FairSteer): a linear 'will tip' probe at the decision position gates a di… [gap] | FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering (2504.14492) | 2025-04 (a… | 5,7 | exists, partly accurate | no | The headline metric is within-set consistency (same job-CV pair across all values of one attribute), but FairSteer's gate is a per-prompt decision. Whenever the… |
| 134 | QeRL Adaptive Quantization Noise (AQN) as an exploration device for LoRA RL | QeRL: Beyond Efficiency -- Quantization-enhanced Reinforcement Learning for LLMs (2510.11696) | 2025-10 | 2 | exists, partly accurate | no | The saturation AQN is supposed to cure is not an exploration deficit, and the published mechanism is off when the saturation happens. The action space here is o… |
| 135 | RLBFF binary-principle reward model (Yes/No entailment logit difference) for the feedback field | RLBFF: Binary Flexible Feedback to bridge between Human Feedback & Verifiable Rewards (2509.21319) | 2025-09 | 4,5 | exists, mechanism accurate | no | The quantity the principle RM would reward is already saturated: on the 31,050-prompt audit, the project's own lexical detector (eval/metrics.py::mentions_attri… |
| 136 | Rubric / principle-following generative reward models as softer targets (RaR, RLBFF-Nemotron GenRM, J1 pairwis… | Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains (2507.17746) | 2025-07 (v… | 4,5,8 | exists, mechanism accurate | no | The benchmark measures only the decision token, and a rubric/GenRM grades the 14-27-word feedback that is generated after and conditioned on that decision. With… |
| 137 | Trajectory extrapolation: ExPO and rank-1 RLVR trajectory extrapolation (RELEX) | You Only Need Minimal RLVR Training: Extrapolating LLMs via Rank-1 Trajectories (2605.21468) | 2026-05 | 2,7 | exists, partly accurate | no | RELEX's premise, a trajectory that is a line with linearly growing projection, is already falsified on GRPO-v3-all's own saved adapters (CPU check of the 20 che… |
| 138 | WAPO: Winner Advantage Policy Optimization (positive-advantage-only clipped policy gradient) | A Gradient Perspective on RLVR Stability and Winner Advantage Policy Optimization (2606.16154) | 2026-06-15… | 2,7 | unverified | unverified | — |
| 139 | LUSPO: Length-Unbiased Sequence Policy Optimization (TRL loss_type='luspo') | Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Va… (2602.05261) | 2026-02 | other | exists, mechanism accurate | no | In this project's regime LUSPO is mathematically the loss already in use. TRL 1.15 implements it as loss = mean_i[ \|y_i\| * (-s_i * A_i) ] with s_i = exp(mean_… |
| 140 | LUSPO: length-unbiased GSPO (per-sequence loss multiplied by \|y_i\|) [gap] | Length-Unbiased Sequence Policy Optimization: Revealing and Controlling Response Length Va… (2602.05261) | 2026-02 | other | exists, mechanism accurate | no | In the current recipe LUSPO is mathematically a no-op, and in the DAPO variant it is a different clipping scheme, not a length fix. With one optimizer step per… |
| 141 | Multi-Reward GRPO with a DeBERTa neutrality classifier plus composite form rewards [gap] | Multi-Reward GRPO Fine-Tuning for De-biasing Large Language Models: A Study Based on Chine… (2511.06023) | 2025-11 | 8 | exists, mechanism accurate | no | The fairness reward is provably inert here and the only live component points at the wrong target. R_fair = P(neutral) is evaluated on a single response, but th… |
| 142 | Positive/negative sample re-weighting (W-REINFORCE lambda=0.1; Asymmetric REINFORCE baseline V): mostly inert… | The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning (lead work; compan… (2506.01347) | 2025-06 | 2,4 | exists, partly accurate | no | Every completion opens with the forced prefix {"decision": " (scripts/grpo_signal_by_reference.py docstring: 99.98% of samples), so the verdict is ONE token in… |
| 143 | Qwen3.5's documented small-model recipe: large-scale RL across environments on an asynchronous, disaggregated… [gap] | Qwen3.5: Towards Native Multimodal Agents (blog; citation block variant: "Qwen3.5: Acceler… | 2026-02 | other | exists, mechanism accurate | no | There is no method here to test. The source is a marketing blog and a model card that describe infrastructure (asynchronous, disaggregated RL with bounded gradi… |
| 144 | RLP: Reinforcement as a Pretraining objective (information-gain reward for a sampled thought) | RLP: Reinforcement as a Pretraining Objective (2510.01265) | 2025-09 (v… | other | exists, partly accurate | no | The adapted reward is a self-rewarding persuasion signal, not a verifier: it pays the policy for feedback text that moves the frozen SFT-v3 (the biased baseline… |
| 145 | Reference-policy resets or mixup (ProRL hard reset; TR-DPO `sync_ref_model`): available, but here the referenc… | ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Mod… (2505.24864) | 2025-05 (P… | 4,7 | exists, partly accurate | no | The method's premise is absent here and its only real effect points the wrong way. ProRL/TR-DPO reset or mix the reference because the KL term grows until it do… |
| 146 | Single-view self-consistency rewards on the attributed prompt (TTRL majority vote, RLSC self-confidence, Intui… | TTRL: Test-Time Reinforcement Learning (2504.16084) | 2025-04 (v… | 2,4 | exists, partly accurate | no | The benchmark decodes greedily (temperature 0.0 in configs/audit/*.yaml), and the majority of 16 temperature-1 rollouts on an attributed prompt is just an estim… |
| 147 | VESPO: Variational Sequence-Level Soft Policy Optimization (TRL loss_type='vespo') and other off-policy / trai… | VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Train… (2602.10693) | VESPO 2026… | 7,8 | exists, partly accurate | no | VESPO/Mu-GRPO/IcePop/TIS all correct a divergence between the sampling distribution and the policy being updated, and in this project that divergence is already… |
