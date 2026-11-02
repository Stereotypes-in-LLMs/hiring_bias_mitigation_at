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
| 10 | 5233609 | G GRPO student s43 audit (nid005906): −0.91 [−1.72, −0.13] pp against SFT-v3 ("better, pending a second seed"), greedy TPR +0.78, agreement +0.04, every guardrail passes, conforms; merge kept 95.1%. Head to head against its parent GRPO + decay R4 s43 at 23.09%: −0.49 [−1.19, +0.21] | 1.24 | 27.68 |
| 11 | 5233661 | I seed 42 audit (nid005576): −0.59 [−1.39, +0.26] pp against SFT-v3 (one epoch: −0.42), greedy TPR +1.29, agreement +0.11, every guardrail passes, conforms; merge kept 95.9%. Head to head against one epoch s42 at 23.13%: −0.21 [−0.69, +0.19] | 1.19 | 28.87 |
| 12 | 5233403 | I seed 43 training (nid005676): 682 steps, 2:30:54; 10.041 GPU-h measured; train_loss 0.028 over the run, attributed KL at the end 0.0030 | 10.06 | 38.93 |
| 13 | 5233871 | I seed 43 audit (nid005514): −0.84 [−1.67, −0.04] pp against SFT-v3 (one epoch: −0.59 [−1.24, −0.03]), greedy TPR +1.20, agreement +0.16, every guardrail passes, conforms; merge kept 95.8%. Head to head against one epoch s43 at 23.08%: −0.30 [−0.78, +0.14] | 1.20 | 40.13 |
| 14 | 5233658 | I seed 44 training (nid005408): 682 steps, 2:33:53; 10.240 GPU-h measured; train_loss 0.028 over the run, attributed KL at the end 0.0040 | 10.26 | 50.39 |
| 15 | 5234047 | I seed 44 audit (nid006514): −0.72 [−1.56, +0.13] pp against SFT-v3 (one epoch: −0.70 [−1.52, +0.08]), greedy TPR +1.04, agreement +0.03, every guardrail passes, conforms; merge kept 95.9%. Head to head against one epoch s44 at 22.96%: −0.04 [−0.54, +0.40] | 1.22 | 51.61 |

## 3. Result

### G. The students on the parents' seed-43 models (DPO: against; GRPO: not better, no seed added)

Head to head against the parent at the student's hire rate (`compare-arms`); seed 42 from
`PATCHING_DISTILL.md`, section 3B. Every audit conforms and passes every guardrail.

| student | seed 42 vs parent | seed 43 vs parent | mean (of the unrounded values) | seed 43 vs SFT-v3 (seed 42) |
|---|---|---|---|---|
| on DPO_V3_4880 | −0.07 [−0.44, +0.33] | +0.58 [−0.22, +1.37] | +0.25 | +0.04 [−1.00, +1.08] (−1.50) |
| on GRPO + decay R4 | +0.02 [−0.77, +0.82] | −0.49 [−1.19, +0.21] | −0.23 | −0.91 [−1.72, −0.13] (−0.16) |

- **Calls (section 1G).**
  - DPO student: **against**, the two-seed mean is ≥ 0.
  - GRPO student: **not better**. Not both seeds are below 0, so no seed 44 is trained without the
    user.
- **Fact.** Against SFT-v3, each student follows its parent's draw:
  - the DPO student was −1.50 on the best DPO seed and +0.04 on seed 43 (parent −0.09);
  - the GRPO student was −0.16 on seed 42 (parent −0.73) and −0.91 on seed 43 (parent −0.44).
- The GRPO student s43's "better, pending a second seed" against SFT-v3 is one of two draws, and
  its seed-42 counterpart was not better. As registered, no student is reported as "distilled
  DPO / GRPO beats SFT-v3".

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

### I. Two epochs (registered call: helps, by a narrow margin; not clearly better than SFT-v3)

`decision_distill_e2_seed{42,43,44}`: 682 steps, the decay over the last 178. Head to head against
one epoch of the same seed at the two-epoch arm's hire rate (`compare-arms`). Every audit
conforms and passes every guardrail against SFT-v3; the merge kept 95.8-95.9%.

| seed | two epochs vs one epoch, pp [95% CI] | two epochs vs SFT-v3, pp [95% CI] | one epoch vs SFT-v3, pp | TPR / agreement vs SFT-v3, pp |
|---|---|---|---|---|
| 42 | −0.21 [−0.69, +0.19] | −0.59 [−1.39, +0.26] | −0.42 | +1.29 / +0.11 |
| 43 | −0.30 [−0.78, +0.14] | −0.84 [−1.67, −0.04] | −0.59 | +1.20 / +0.16 |
| 44 | −0.04 [−0.54, +0.40] | −0.72 [−1.56, +0.13] | −0.70 | +1.04 / +0.03 |
| mean | **−0.18** | −0.72 | −0.57 | |

- **Call (section 1I).** Every seed is below 0 and the mean, −0.185, is at most −0.15: by the
  registered rule a second epoch **helps**. Every head-to-head interval holds 0, and the mean
  clears the threshold by 0.035 pp; without seed 43 it would be −0.13, undecided.
- **G3 against SFT-v3.** Every seed is below 0, but only seed 43's interval is, so two epochs are
  **not "clearly better than SFT-v3"**. The standard's verdicts: s43 "better, pending a second
  seed"; s42 and s44 "not better: the interval reaches 0".
- **Fact.** Seed 44 gained nothing from the second epoch (−0.70 to −0.72); seeds 42 and 43 moved
  by −0.17 and −0.25. Training cost doubled: 10.0-10.2 GPU-h a seed against 5.1-5.3 (`reports/cost/steps.csv`, jobs 5162175-5162177).
- **Interpretation, not confirmed.** The attributed KL kept falling (end of run 0.003-0.006
  against 0.013-0.023 after one epoch), and the matched metric moved a little in the same
  direction. One epoch was close to, but not at, the ceiling this objective reaches on these
  data.

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

The round closed on 2026-11-02 at the user's request ("收尾"), at 51.61 of 100 GPU-h allocated
(ledger, row 15). No seed-44 student was trained.

**What this round's evidence supports.**
- Distilling a DPO or GRPO model does not make it better than its parent (G). Against SFT-v3
  each student follows its parent's seed draw. There is no evidence for "distilled DPO / GRPO
  beats SFT-v3".
- Layers 15-20 alone do not carry distillation's gain (H). With section F of
  [`PATCHING_DISTILL.md`](PATCHING_DISTILL.md) (the window removed keeps the gain), the change
  that matters lies mostly outside the window that patching locates. Patching describes where
  the attribute travels; it has not yet told distillation where to act.
- A second epoch moves the matched metric a little further in the direction of the first
  (I, "helps" by the registered rule), at twice the training cost, and two epochs are still
  not clearly better than SFT-v3.
- Distillation's remaining instability is largely the same borderline sets on every seed (J);
  DPO's and GRPO's is a different draw per seed.

**What it does not support.**
- Two epochs as the new default. The registered call is "helps", but by 0.035 pp past the bar,
  with every head-to-head interval holding 0 and seed 44 unchanged. My recommendation is to
  keep one epoch as the recorded arm and report two epochs as a sensitivity check. The user
  decides.
- Any patching-guided distillation claim, so the poster subtitle stays as it is.

**Explanations for I, and the cheapest check that separates them.**
- The second epoch compresses margins further on the borderline pairs, as the first epoch did
  (`PATCHING_DISTILL.md`, scale-only control). If so, the fixed sets are on the 35 borderline
  pairs and the margin spread there is narrower than after one epoch.
- Seed noise: three draws within ±0.3 pp of one epoch, each interval holding 0.
- Check: `seed_overlap.py`'s share-on-borderline and the margin spread, for the two-epoch
  adapters against one epoch. CPU, minutes, on margin files already written. Not run.

**An alternative to more training.** Distillation's remaining unstable sets sit almost entirely
on 35 pairs close to SFT-v3's cut (J). Reading those pairs, and GPT-4o's labels on them, may say
more about what is left than another training arm: some may be cases where hire and reject are
both defensible.

## 5. Caveats

- Every verdict is exploratory: the 450-pair benchmark has been read many times (section 1).
- I's "helps" rests on the three-seed mean. Without seed 43 the mean is −0.13, "undecided".
- G's GRPO student has two seeds, and the second-seed rule did not trigger. Its seed-43
  "better, pending a second seed" against SFT-v3 is one draw.
- H ablates adapters trained with every layer present. It does not show what a model trained
  without those layers would do; section A of `PATCHING_DISTILL.md` covers that case.
- GPU-h in the ledger are allocated (wall time × 4 GPUs). Measured training figures come from
  the `cost` records named in each row. Rows 1-2 are jobs that failed on faulty nodes
  (nid005679, nid006377) and are charged in full. Later jobs exclude both nodes.
- Code versions: training ran from dce86f0, dda1c38, 4a17318, 5f40053 and 545cc51. The audits
  ran from dda1c38 to 399a1f8 (the first line of each job log under `logs/`). Between
  dce86f0 and 399a1f8 the only change under `scripts/`, `src/` or `configs/` is the new
  `scripts/seed_overlap.py` (dda1c38). `compare-arms` JSONs record `code.commit` and
  `code.uncommitted: []`.

## 6. Reproducing

Students (G), on a parent's merged checkpoint, then two epochs (I):

```bash
HBM_COST_ARM=distill_followup REPO=$PWD CONFIG=configs/mitigation/distill/decision_distill_grpo_r4_seed43.yaml \
  scripts/cscs/submit.sh -p normal -t 04:00:00 --exclude=nid005679,nid006377 scripts/cscs/train_distill.sbatch
HBM_COST_ARM=distill_followup REPO=$PWD \
  AUDIT=configs/audit/qwen3.5-9b_en_decision_distill_e2_seed43_gh200.yaml \
  AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh -t 00:30:00 --exclude=nid005679,nid006377 scripts/cscs/evaluate_arm.sbatch
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw OPENBLAS_NUM_THREADS=1 \
  ../venv/bin/python scripts/operating_point_control.py compare-arms \
  --config configs/audit/qwen3.5-9b_en_decision_distill_e2_seed43_gh200.yaml \
  --against configs/audit/qwen3.5-9b_en_decision_distill_seed43_gh200.yaml
OPENBLAS_NUM_THREADS=1 ../venv/bin/python scripts/seed_overlap.py     # J, CPU
```

The window kept alone (H):

```bash
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
