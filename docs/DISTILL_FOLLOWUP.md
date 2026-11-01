# Follow-ups to decision distillation: second seeds of the students, the window kept alone, two epochs

Started 2026-11-01 at the user's request, after the patching-informed round
([`PATCHING_DISTILL.md`](PATCHING_DISTILL.md)): "B2 run now … budget 100 GPU-h … start the
candidates, three at a time". Branch `xinyang/distill-followup` (from main ddebd82). The
comparison is [`EVAL_STANDARD.md`](EVAL_STANDARD.md), unchanged; the reference arm is SFT-v3.

## 0. Decisions recorded on 2026-11-01 (the user)

- **Decision distillation stays recorded as better than SFT-v3** (the 2026-10-27 reading of rule
  4, [`DECISION_DISTILL.md`](DECISION_DISTILL.md)). The generated verdicts are unchanged.
- **GRPO + decay R3 stays recorded as better** (its standard verdict stays "not better: the
  interval reaches 0", −0.71 [−1.89, +0.39]).
- **The poster quotes the common point**, every arm moved to SFT-v3's own hire rate of 22.51%
  ([`FINAL_TABLE.md`](FINAL_TABLE.md)), instead of each arm's own point.
- **Budget for this round: 100 GPU-h allocated**, under the cost arm `distill_followup`. Up to
  three jobs at a time.

## 1. Registration (2026-11-01, before any job)

Every arm is audited with `evaluate_arm.sbatch` against SFT-v3, then compared head to head with
`operating_point_control.py compare-arms` (CPU) against its control, at the arm's matched hire
rate. The final adapter of each trained arm is declared now; no checkpoint is chosen. The
450-pair benchmark has been read many times, so every verdict here is exploratory.

### G. The students on the parents' seed-43 models

**Question.** Does decision distillation improve a DPO or GRPO model on a second draw of the
parent? The seed-42 students were not better than their parents (DPO −0.07 [−0.44, +0.33], GRPO
+0.02 [−0.77, +0.82]), and both parents were the best of their method's three seeds, so part of
any change there could be regression to the method mean.

**Arms.** `decision_distill_dpo4880_seed43` (parent DPO_V3_4880 s43, its audited
checkpoint-305) and `decision_distill_grpo_r4_seed43` (parent GRPO + decay R4 s43, its audited
final adapter), training seed 43; everything else as the seed-42 students.

**Control.** The parent at the student's hire rate: runs
`Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_seed43_step305` and
`Qwen3.5-9B--en--grpo--adapter--en_only_grpo_improve_r4_tipped_seed43`.

**Rule, per method, on its two seeds (42 and 43).**
- Better than its parent: both changes below 0 and at least one interval below 0. A seed-44
  student is then trained (registered now, same rule on three seeds).
- Against: the mean of the two changes ≥ 0.
- Otherwise not better, and no seed is added without the user.
- Against SFT-v3 the students are context only and never reported as "distilled DPO / GRPO beats
  SFT-v3".

### H. The window kept alone (the complement of section F)

**Question.** F showed that the trained all-layer adapters keep their gain with layers 15-20
removed (+0.11 pp head to head). Does the window's part of the same adapters carry the gain on
its own, i.e. does it duplicate the rest?

**Arms.** `decision_distill_seed{42,43,44}_xk15_20`: the trained all-layer adapters with
`lora_B` zeroed in every layer outside 15-20 (`scripts/ablate_adapter_layers.py --keep`). Not
trained; three audits.

**Control.** The full adapter of the same seed, `decision_distill_seed{42,43,44}`.

**Rule, on the three seeds' head-to-head changes.**
- The window part carries the gain on its own: mean ≤ +0.20 pp and every seed ≤ +0.30.
- It does not: mean > +0.30.
- Otherwise undecided.
- **Expected** from A (window trained alone, +0.53) and F: it does not.

Like F, this ablates adapters trained with every layer present. It does not show what a model
trained on the window would do.

### I. Two epochs

**Question.** Is distillation's gain at its ceiling after one epoch
([`DECISION_DISTILL.md`](DECISION_DISTILL.md), open question 2: the attributed KL is still
falling at the decay, 0.013-0.023 at the end)?

**Arms.** `decision_distill_e2_seed{42,43,44}`: `decision_distill_seed{S}` with
`num_train_epochs: 2` (682 optimizer steps) and the linear decay over the last 178 steps, the
same 26% share as one epoch's 89 of 341. Nothing else changes.

**Control.** One epoch of the same seed, `decision_distill_seed{42,43,44}`.

**Rule, on the three seeds' head-to-head changes.**
- Helps: every seed below 0 and mean ≤ −0.15 pp.
- Does not help: mean ≥ 0.
- Otherwise undecided.
- Against SFT-v3 the day's goal G3 carries over: every seed below 0 and at least two intervals
  below 0 is "clearly better than SFT-v3" (exploratory).
- A failed guardrail against SFT-v3 counts against the arm whatever the change.

### J. CPU check: the overlap of DPO_V3_4880's unstable sets across seeds

The cheapest check named in [`DPO_V3_4880_SEEDS.md`](DPO_V3_4880_SEEDS.md) and
[`FINAL_TABLE.md`](FINAL_TABLE.md): the overlap of the sets each seed leaves unstable at its
own point, on the margin files already written. It is descriptive and decides nothing.

### Order, cost, stop conditions

- **Order.** Three slots:
  - I s42 training and G DPO s43 training start first;
  - the third slot runs H's three audits, then G GRPO s43 training;
  - the students' audits, I s43, I s44 and I's audits follow as slots free.
- **Estimate (allocated GPU-h).**

  | arm | per job | total |
  |---|---|---|
  | G | training about 5.6, audit 1.2 | about 13.6 for both |
  | H | audit about 1.2 | about 3.7 |
  | I | training about 11 (two epochs), audit 1.2 | about 36.6 |
  | **round** | | **about 54** |

  A seed-44 student pair, if G triggers it, adds about 13.6.
- **Stop.** Stop and ask before the round would pass 100 GPU-h. A failed job is resubmitted at
  most twice. Training stops itself under `memory_floor_gb`. No change to the evaluation, the
  data, the split or `reports/eval_scope.yaml`.
- **Conformance.** No tracked file under `scripts/`, `src/` or `configs/` of this checkout is
  edited while an audit runs from it.

## 2. Ledger

| # | job | what | GPU-h allocated | running total |
|---|---|---|---|---|
| 1 | 5232809 | I seed 42 training: failed at start on nid005679, no GPU visible to the job ("0 x NVIDIA GH200", the trainer: "doesn't support bf16/gpu", `logs/xw-distill-5232809.out`); resubmitted excluding nid005679 and nid006377 | 0.56 | 0.56 |
| 2 | 5232811 | H seed 42 audit: failed on nid006377, "No CUDA GPUs are available" (the node of PATCHING_DISTILL ledger row 15, `logs/xw-eval-arm-5232811.out`); resubmitted with the same exclusion | 0.57 | 1.13 |
| 3 | 5232839 | H seed 42 audit (retry 1, nid005561): −0.17 pp against SFT-v3, every guardrail passes, conforms; merge kept 91.6%. Head to head against the full adapter s42 (`compare-arms`, CPU): +0.22 [−0.50, +0.97] | 1.00 | 2.13 |
| 4 | 5232894 | H seed 43 audit (nid006478): −0.20 pp against SFT-v3, every guardrail passes, conforms; merge kept 91.6%. Head to head against the full adapter s43: +0.33 [−0.18, +0.86] | 1.21 | 3.34 |
| 5 | 5233216 | H seed 44 audit (nid005651): −0.02 pp against SFT-v3, every guardrail passes, conforms; merge kept 91.8%. Head to head against the full adapter s44: +0.67 [−0.11, +1.47] | 1.18 | 4.52 |
| 6 | 5232810 | G DPO student s43 training (nid006002): 341 steps, 1:19:15; trained on DPO_V3_4880 s43 checkpoint-305 merged (`run_config.json`); 5.262 GPU-h measured | 5.28 | 9.80 |
| 7 | 5233351 | G DPO student s43 audit (nid005942): +0.04 [−1.00, +1.08] pp against SFT-v3, greedy TPR −4.49, agreement −0.90, every guardrail passes, conforms; merge kept 96.6%. Head to head against its parent DPO_V3_4880 s43 at 20.69%: +0.58 [−0.22, +1.37] | 1.22 | 11.02 |
| 8 | 5233279 | G GRPO student s43 training (nid005933): 341 steps, 1:19:48; trained on GRPO + decay R4 s43 merged; 5.301 GPU-h measured | 5.32 | 16.34 |
| 9 | 5232838 | I seed 42 training (retry 1, nid005944): 682 steps, 2:31:33; 10.085 GPU-h measured; train_loss 0.028 over the run, attributed KL at the end 0.0063 | 10.10 | 26.44 |

## 3. Result

### H. The window kept alone (registered call: does not carry the gain on its own)

`decision_distill_seed{42,43,44}_xk15_20`: the trained all-layer adapters with `lora_B` zeroed
outside layers 15-20 (26 of 128 modules kept). Every audit conforms and passes every guardrail;
the merge kept 91.6-91.8%.

| seed | against the full adapter (`compare-arms`), pp [95% CI] | against SFT-v3, pp | the full adapter against SFT-v3, pp |
|---|---|---|---|
| 42 | +0.22 [−0.50, +0.97] | −0.17 | −0.42 |
| 43 | +0.33 [−0.18, +0.86] | −0.20 | −0.59 |
| 44 | +0.67 [−0.11, +1.47] | −0.02 | −0.70 |
| mean | **+0.41** | −0.13 | −0.57 |

- **Call.** The mean is above +0.30, so by the rule registered in section 1H the window's part of
  the trained adapters does not carry the gain on its own. Every interval holds 0; the call rests
  on the three-seed mean, as registered.
- **Fact, with section F of `PATCHING_DISTILL.md`.** On the same three adapters, removing layers
  15-20 kept the gain (+0.11 against the full adapter) and keeping only layers 15-20 loses most of
  it (+0.41). Trained alone at the same learning rate, the window lost it too (A, +0.53).
- **Interpretation, not confirmed.** Distillation's useful change lies mostly outside the window
  that patching locates. The window's part does not duplicate the rest. This is an ablation of
  adapters trained with every layer present, so it does not show what a model trained without
  those layers would do.

### J. Overlap of the unstable sets across training seeds (CPU, descriptive)

`scripts/seed_overlap.py`, `reports/seed_overlap/overlap.csv`: unstable within-group sets at each
seed's own greedy cut (of 2,700), op-margin-v2 margins. "SFT-v3" is SFT-v3 moved to the same two
hire rates (weighted where its cut ties), the overlap the operating points alone would give.

| recipe | unstable sets, s42 / s43 / s44 | Jaccard of seed pairs 42-43 / 42-44 / 43-44 | SFT-v3 at the same rates | on the 35 borderline pairs |
|---|---|---|---|---|
| DPO_V3_4880 | 25 / 46 / 57 | 0.15 / 0.14 / 0.26 | 0.22 / 0.19 / 0.83 | 80 / 74 / 81% |
| GRPO + decay R4 | 38 / 45 / 51 | 0.14 / 0.09 / 0.28 | 0.10 / 0.09 / 0.92 | 68 / 91 / 80% |
| Decision distillation | 46 / 38 / 39 | 0.53 / 0.60 / 0.45 | 0.83 / 0.97 / 0.86 | 96 / 97 / 95% |

- **Fact.** DPO's and GRPO's seeds leave mostly different sets unstable: at most about a quarter
  of the union is shared. Where two seeds sit at nearly the same hire rate (43 and 44 of either
  method), SFT-v3 at those rates shares 83-92% and the seeds 26-28%.
- **Fact.** Distillation's seeds share about half of their unstable sets, and 95-97% of those
  sets lie on SFT-v3's 35 borderline pairs (DPO 74-81%, GRPO 68-91%).
- **Interpretation, not confirmed.** DPO's seed spread (sd 0.65 pp) fits a training draw that
  decides which sets end up straddling the widened cut, not one fixed group of hard sets; DPO
  s42's low count would then be a favourable draw. Distillation's remaining instability is
  largely the same borderline sets on every seed.

## 4. Reading

## 5. Caveats

## 6. Reproducing

```bash
HBM_COST_ARM=distill_followup REPO=$PWD CONFIG=configs/mitigation/distill/decision_distill_e2_seed42.yaml \
  scripts/cscs/submit.sh -p normal -t 04:00:00 scripts/cscs/train_distill.sbatch
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw ../venv/bin/python scripts/ablate_adapter_layers.py \
  --adapter outputs/distill/decision_distill_seed42 --layers 15 20 --keep \
  --out outputs/distill/decision_distill_seed42_xk15_20
HBM_COST_ARM=distill_followup REPO=$PWD \
  AUDIT=configs/audit/qwen3.5-9b_en_decision_distill_seed42_xk15_20_gh200.yaml \
  AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh -t 00:30:00 scripts/cscs/evaluate_arm.sbatch
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw OPENBLAS_NUM_THREADS=1 \
  ../venv/bin/python scripts/operating_point_control.py compare-arms \
  --config configs/audit/qwen3.5-9b_en_decision_distill_seed42_xk15_20_gh200.yaml \
  --against configs/audit/qwen3.5-9b_en_decision_distill_seed42_gh200.yaml
```
