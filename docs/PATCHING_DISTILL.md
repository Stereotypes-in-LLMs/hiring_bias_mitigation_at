# Patching-informed decision distillation, and distillation on the DPO and GRPO models

Asked by the user on 2026-11-01 ("Start the runs"): distillation with the LoRA on Marta's pathway
layers for seeds 42-44, and decision distillation on top of DPO_V3_4880 s42 and GRPO + lr decay
R4 s42, with a cap of 40 GPU-h and one job at a time. Branch `xinyang/patching-distill` (from
main d8204e4). The comparison is [`EVAL_STANDARD.md`](EVAL_STANDARD.md), unchanged; the
reference arm is SFT-v3. Costs under `HBM_COST_ARM=patching_distill`.

**Why this round.** Decision distillation ([`DECISION_DISTILL.md`](DECISION_DISTILL.md)) was
chosen from the self-distillation family of the method survey, before any patching of a
distilled model. The poster's story, "mechanistic findings inform distillation", is therefore
not true of it. The one patching-guided variant so far, the LoRA on layers 0-20
([`DISTILL_LAYERS.md`](DISTILL_LAYERS.md)), was chosen from a masked check on the distilled model
itself; against all layers it came out at +0.296 pp [+0.03, +0.62] (seed 43), in its registered
undecided band, just short of "gain lost". This round tests a
design taken from Marta Samoilenko's patching of SFT-v3, and fills the poster's two empty
student slots.

## 0. The goal for 2026-11-01, and the rules for exploring

The user, 2026-11-01: "set a goal to make the results better; today you can keep exploring", up
to three jobs at a time today.

**Goal.** Make "patching informs distillation" true and the result stronger, in this order:
- **G1 (registered, section 1A).** Distillation confined to the pathway window, with a fifth of
  the trainable parameters, is not worse than all-layer distillation (H1).
- **G2.** Distilling the DPO and GRPO models, each with a window taken from that parent's own
  patching, improves on the parent at matched hire rate.
- **G3 (exploration).** A distillation variant whose three-seed mean is clearly better than
  SFT-v3: every seed below 0 and at least two seeds with the 95% interval below 0.

**Rules.**
- Each variant is written into this card, with its seeds and decision rule, before its job.
- Nothing is called better on one seed; a promising single seed is followed by two more seeds.
- Every variant tried is reported, including the worse ones.
- The 450-pair benchmark has been read many times, so the verdicts stay exploratory.
- Cap for the day: 60 GPU-h allocated under `patching_distill` (the 40 registered plus 20 for
  exploration). Stop and ask before going over.

## 1. Registration (2026-11-01, before any run)

### A. Distillation on the pathway Marta's patching found

**Where the window comes from** ([`INTERP_SFT_V3.md`](INTERP_SFT_V3.md), Patching, 64 identical
pairs per contrast, three contrasts). In the base model the attribute effect reaches the
decision tokens between layers 15 and 20: the share patched at the decision tokens rises from
0.63-0.74 by layer 18 to 0.92-0.95 by layer 20. Full-attention layer 15 is the hub, with head
L15.H11 the largest verified head, and the readout runs through MLP 31. SFT-v3 removes 61-77% of
the per-pair effect, and what is left takes the same route: 0.49-0.56 by layer 18, 0.87-0.88 by
layer 20, L15.H11 still the largest head in all three contrasts.

**Question.** Is the distillation update enough when it is confined to the layers where the
attribute effect reaches the decision tokens?

**Change (one variable).** `lora_layer_range: [15, 20]`. That keeps 26 of the 128 LoRA modules:
the six MLPs of layers 15-20 and the two full-attention layers in that window, 15 and 19 (the
linear-attention layers carry no target module). From the per-module sizes in
`reports/cost/params.csv` (an MLP 1,572,864 and a full-attention layer 983,040 trainable
parameters), that is 11,403,264 trainable parameters, 0.127% of the 8,953,803,264-parameter text
backbone, against 58,195,968 (0.650%) for all layers. The training job records the exact count.
The readout, MLP 31, is outside the window and stays frozen: the question is whether changing
where the attribute is written to the decision position is enough.

Everything else matches `decision_distill_seed{42,43,44}`: SFT-v3 merged plus a fresh LoRA r32
α64, the same 10,907 rows (`configs/data/en_only_distill_v3_all.yaml`), the same KL on the
decision token to SFT-v3 on the attribute-free CV, lr 3e-5 with the same warmup-stable-decay
schedule, one epoch (341 optimizer steps, as the controls ran: `reports/cost/steps.csv`), the
same seeds. Configs
`configs/mitigation/distill/decision_distill_l15_20_seed{42,43,44}.yaml`.

**Control.** The all-layer arm of the same seed: `decision_distill_seed{42,43,44}`, audited
already (−0.42, −0.59, −0.70 pp against SFT-v3).

**Hypotheses, with the call fixed now.** Δ_k is the matched change of the pathway arm against
the all-layer arm of seed k (`compare-arms`); Δ is their mean over the three seeds.
- **H1, the pathway is enough:** Δ ≤ +0.20 pp, and every seed passes every guardrail against
  SFT-v3 with a matched change below 0.
- **H0, the pathway loses the gain:** Δ > +0.30 pp.
- Otherwise: undecided.

Each seed also gets the standard's verdict against SFT-v3, as every arm does. The prior evidence
leans against H1: confining the update to layers 0-20 already cost +0.296 pp on one seed.

H0 cannot tell "the window is not enough" from "an adapter five times smaller underfits at the
same learning rate": the learning rate is deliberately not retuned, so the only change is the
window. A result under H0 is worded "distillation confined to layers 15-20 at the same learning
rate", not "the pathway is not enough".

**What the poster may say, fixed now.** Only this arm is designed from the patching result. Under
H1 the subtitle may say the mechanistic finding informs distillation. Under H0 or undecided the
poster reports it as a patching-informed variant that was worse, or undecided, and the subtitle
stays as it is. Either way the arm is reported.

### B. The same distillation on the DPO and GRPO models

**Question.** Does decision distillation, unchanged (all layers), improve the two other
post-SFT methods' models when each is its own teacher?

**Arms.** `decision_distill_dpo4880_seed42`: parent DPO_V3_4880 seed 42, its audited
checkpoint-305. `decision_distill_grpo_r4_seed42`: parent GRPO + lr decay R4 seed 42, its audited
final adapter. Training starts from the parent merged (its adapter chain SFT-v3 then DPO or
GRPO); the teacher is the same weights with the LoRA disabled, so the parent itself, on the
pair's attribute-free CV. Everything else matches `decision_distill_seed42`.

**Comparisons.** The question is answered by `compare-arms` against the parent: runs
`Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305` and
`Qwen3.5-9B--en--grpo--adapter--en_only_grpo_improve_r4_tipped_v2`. The standard's verdict
against SFT-v3 is context: it carries the parent's own seed-42 draw, and DPO's did not replicate
(seed 42 −1.03, three-seed mean −0.31, [`DPO_V3_4880_SEEDS.md`](DPO_V3_4880_SEEDS.md); GRPO R4's
seeds −0.73 / −0.44 / −0.16). A student's result is never reported as "distilled DPO / GRPO beats
SFT-v3". One seed each, so a pilot: a change against the parent is reported with its interval,
and even an interval below 0 reads "better than its parent, pending a second seed".

**Expected if distillation transfers.** The change against the parent is below 0, as it was for
SFT-v3 (−0.57 pp, three seeds), and every guardrail passes against SFT-v3.
**Against.** A change against the parent ≥ 0, or a failed guardrail against SFT-v3.

**Added 2026-11-01, before either student job.** Both parents are the best of their method's
three seeds: DPO s42 −1.03 against −0.09 and +0.20; GRPO R4 s42 −0.73 against −0.44 and −0.16
(`reports/reference_metrics.csv`). A change ≥ 0 against such a parent therefore reads "not
distinguishable from regression to the method mean". Each student is also reported against
SFT-v3 next to its method's three-seed mean (DPO −0.31, GRPO R4 −0.44). A windowed student (its
LoRA confined to a window read from that parent's patching) is trained only if the all-layer
student is better than its parent, upper bound below 0.

### C. The whole pathway window, GDN mixers included (registered 2026-11-01, after A seed 42)

**Why.** A seed 42 came out +0.13 pp against SFT-v3 and +0.26 [−0.32, +0.82] against all
layers, so H1 can no longer be reached (it needs every seed below SFT-v3). Arm A adapted only the
MLPs and the two full-attention layers of the window: the default LoRA targets have no
linear-attention module, so the GDN mixers of layers 16, 17, 18 and 20 stayed frozen. In SFT-v3
they carry part of the arrival: at the post-attribute tokens GDN 16-18 take 0.13-0.41 of the
effect, and the layer-17 step at the decision tokens is mostly its GDN mixer
(`reports/interp/patching/sft_v3-grid-*-sel_sft_v3/real.csv`). Against this reading, the largest
carrier at the decision tokens, attention 19, was trainable in arm A, and GDN 17 reads a residual
that the trainable layers write; the training logs show no underfitting. This arm is designed
after seeing A seed 42, so it is exploratory.

**Change against arm A (one variable).** The LoRA targets in layers 15-20 add the three large GDN
projections, `linear_attn.in_proj_qkv`, `in_proj_z` and `out_proj`: 38 modules, 15,073,280
trainable parameters (0.168% of the text backbone, a quarter of all layers' 58,195,968). Counted
by building the model with empty weights and applying `peft_config` (layers 15-20 only, the same
26 modules as arm A plus 12 GDN projections). Configs
`configs/mitigation/distill/decision_distill_w15_20_seed{42,43,44}.yaml`.

**Pilot rule, fixed now.** Seed 42 first. Seeds 43 and 44 run only if seed 42's change against
the all-layer arm of seed 42 is ≤ +0.20 pp and its change against SFT-v3 is below 0. The
three-seed call then uses arm A's bands: "the window is enough" if the mean change against the
all-layer arm is ≤ +0.20 pp and every seed is below SFT-v3 with every guardrail passing;
"loses the gain" above +0.30; undecided in between. Under the first outcome the poster may say:
distillation confined to the pathway layers, with a quarter of the trainable parameters, is not
worse than distillation on all layers. The served model stays the same 9B model.

### D. The same parameter budget spread over every layer (registered 2026-11-01, after A's call)

**Why.** Arm A reached H0, and its patching shows the confined adapter barely moves SFT-v3
(section 3A). Arm C confines a quarter-size adapter to the pathway window. Arm D is its control
at the same size: LoRA rank 8 instead of 32 on every layer and module of the all-layer arm (alpha
16, so alpha/r stays 2), 128 modules and 14,548,992 trainable parameters (0.162% of the text
backbone; arm C 15,073,280), counted the same way. Configs
`configs/mitigation/distill/decision_distill_r8_seed{42,43,44}.yaml`; everything else matches
`decision_distill_seed{42,43,44}`.

**Question.** At a quarter of the trainable parameters, does placing them on the pathway window
(C) or spreading them over all layers at low rank (D) keep distillation's gain?

**Pilot rule, fixed now (the same as C).** Seed 42 first; seeds 43 and 44 only if seed 42 is
≤ +0.20 pp against the all-layer rank-32 arm of seed 42 and below SFT-v3. Three-seed bands as for
A and C. If C and D both pass their pilots, running both to three seeds would pass the day's
60 GPU-h cap: ask the user first. Under "enough" the poster may say: distillation with a quarter
of the trainable parameters (rank 8, all layers) is not worse than rank 32. The served model is
the same 9B model.

**Added 2026-11-01 15:20, after D's training started and before its audit.** Three secondary
readouts are reported for D beside the gate and decide nothing: the decisions changed against
SFT-v3 (raw parquets; A 171, C 144, all layers 472 at seed 42), the median attribute-free |m|
ratio against SFT-v3 (from the op-margin-v2 margins), and the overlap of fixed sets with the
all-layer arm of seed 42. Their reading thresholds would be post hoc, so none is set.

### E. The teacher's margin floored at 4 logits (registered 2026-11-01 15:25, after A, B and C)

**Why.** What patching and its controls say distillation does, and what it leaves:
- It keeps SFT-v3's pathway and turns the whole readout down, margin scale ×0.45-0.52 with the
  attribute effect shrinking about as much (`DISTILL_PATCHING.md`). A pure rescaling leaves the
  matched statistic at exactly 0, and 83-87% of the sets distillation fixes lie on the 35 pairs
  SFT-v3 decides by less than 2 logits (`SCALE_CONTROL.md`).
- On those pairs its target is soft. 5,894 of the 9,140 attributed training rows (64.5%) are on
  pairs whose SFT-v3 samples split on the attribute-free CV (`distill_train.parquet`,
  `free_hire_rate` strictly between 0 and 1). Near q = 0.5 the two-token KL is flat, so the loss
  does not push a borderline pair's variants to one side of the cut.
- Arms A and C show that moving the update to the pathway window does not help; this arm changes
  what the update is asked to do instead.

The idea is older than this round: a sharpened target or a hinge on the greedy sign is open
question 1 of [`DECISION_DISTILL.md`](DECISION_DISTILL.md). The value F = 4 was picked after a
post hoc reading of the benchmark's margins in the planning review (not committed), so the arm
is exploratory.

**Change against `decision_distill_seed42` (one variable).** `teacher_margin_floor: 4.0`. The
teacher's two-token margin m = z_hire − z_reject on the pair's attribute-free CV becomes
sign(m)·max(|m|, 4) before the softmax, on every row, the attribute-free anchors included: a
target of p(hire) = 0.982 or 0.018 on SFT-v3's own attribute-free side. A margin of at least 4,
and an exact tie, keep today's target. Everything else is unchanged: SFT-v3 merged plus a fresh
LoRA r32 α64 on all 128 modules, the same 10,907 rows, lr 3e-5 WSD, 341 steps, the final adapter.
Code on branch `xinyang/patching-distill-floor` (commit fa0b9f2, worktree `-pfloor`, tests in
`tests/test_decision_distill.py`), merged into this branch before the audit. Configs
`configs/mitigation/distill/decision_distill_floor4_seed{42,43,44}.yaml`.

**Pilot rule, fixed now.** Seed 42. Δ is its matched change against `decision_distill_seed42`
(`compare-arms`), beside the standard's verdict against SFT-v3.
- **Promising:** below 0 against SFT-v3 with every guardrail passing, and Δ ≤ 0. Seeds 43 and 44
  then need the user's budget.
- **Worse:** Δ > +0.20, or a guardrail fails. Stopped and reported as tried.
- Otherwise undecided: reported, and no seed is added without the user.

**Stop conditions.** The job is cancelled if, at its first two logging points (steps 5 and 10),
the loss is not finite or the share of attributed rows the floor lifts (`distill/attr_floored`)
is below 0.2. `memory_floor_gb` as for every arm.

**Secondary readouts, deciding nothing.** Decisions changed against SFT-v3 and against
`decision_distill_seed42`; the median attribute-free |m| ratio against SFT-v3 (op-margin-v2
margins); the unstable sets on the 35 borderline pairs against plain distillation.

**Cost and cap.** About 5.5 GPU-h of training and 1.25 of audit. With D's audit and the three
ablation audits the day would end near 60 GPU-h; if the measured total would pass 60 with this
arm's audit, the audit waits for the user's decision.

### F. The window removed from the trained all-layer adapter (registered 2026-11-01, after A, B and C)

**Why.** Arm A tested whether layers 15-20 are enough: trained alone, at the same learning rate,
they lose the gain (H0, section 3A). It did not test whether the trained all-layer adapter needs
them. In the masked check of seed 43 (`reports/interp/distill/distill43/mask_shares.csv`) the
attribute-specific shrink came more from layers 0-20 than from 21-31. In its residual analysis
(`reports/interp/distill/distill43/residual.csv`; depth d is the residual entering layer d) the
attribute difference against SFT-v3 falls mostly before the window for the two combat contrasts
(ratio 0.98-0.99 at depth 7, 0.88 at depth 15), at the window's first layer for the atheist
contrast (0.94 at depth 15, 0.89 at depth 16), and again at the final depth (0.67-0.71). This
ablation reads the window's part with the standard instead of a margin proxy.

**An ablation, not an arm.** For each seed of `decision_distill_seed{42,43,44}`, `lora_B` of the
26 modules in layers 15-20 (the six MLPs and full-attention layers 15 and 19; 11,403,264 of the
58,195,968 trainable parameters) is set to zero by `scripts/ablate_adapter_layers.py`, and every
other tensor is copied unchanged. No training. Adapters
`outputs/distill/decision_distill_seed{42,43,44}_xw15_20/`, audit configs
`configs/audit/qwen3.5-9b_en_decision_distill_seed{42,43,44}_xw15_20_gh200.yaml`, each audited
with `evaluate_arm.sbatch` against SFT-v3 and then compared with `compare-arms` against the
all-layer arm of the same seed.

**Call, fixed now.** Δ_k is the matched change of the ablated adapter against the full adapter of
seed k; Δ is the mean over the three seeds.
- **Not needed:** Δ ≤ +0.20 pp, and every seed below SFT-v3 with every guardrail passing.
- **Needed:** Δ > +0.30 pp.
- Otherwise undecided.

The remaining 102 modules were trained with the window present, so a loss under "needed" can be
co-adaptation as well as the window's own part, and "not needed" says the rest of the adapter
suffices as trained, not that a model trained without the window would. Wording under "not
needed": "removing layers 15-20 from the trained all-layer adapter keeps the gain"; never
"the window plays no part". Each seed also gets the decisions it changes against the full
adapter (raw parquets).

**Cost.** Three audits, about 1.25 GPU-h each (ledger rows 2, 8, 9, 13, 14). Seeds 42 and 43 first,
seed 44 when a slot frees.

### Evaluation, splits and stop conditions

- The final adapter of each arm, declared now. `evaluate_arm.sbatch` against SFT-v3, then
  `compare-arms` against the control or the parent (CPU).
- No checkpoint is chosen. The training rows contain no benchmark candidate
  (`assert_no_leakage`); the 450-pair audit is the test. As for every post-SFT arm, the benchmark
  has been read many times during development, so verdicts are exploratory
  ([`DECISION_DISTILL.md`](DECISION_DISTILL.md)). The window comes from Marta's patching
  (`reports/interp/latex/activation_patching_report.tex`, findings 1-3) and SFT-v3's patching on
  the grid was read, not tuned on.
- Order: A seed 42 (train, then audit), A seed 43, A seed 44, B on DPO, B on GRPO. One Slurm job
  at a time on the shared account.
- Cap: 40 GPU-h allocated under `patching_distill`. Estimate: training about 5 GPU-h measured
  and 5.6 allocated an arm (decision_distill_seed42 took 5.14 measured, job 5162175), evaluation 1.2
  allocated: about 34 GPU-h for all five. If the cap would be crossed, the remaining arms are
  dropped in reverse order (GRPO student first) and the card says which.
- A job that fails is resubmitted at most twice. Training stops itself under `memory_floor_gb`.
- No change to the evaluation, the data, the split or `reports/eval_scope.yaml`.

## 2. Ledger

| # | job | what | GPU-h allocated | running total |
|---|---|---|---|---|
| 1 | 5227361 | A seed 42 training: 341 steps, 1:13:29; adapter on layers 15-20 only, 26 modules, 11,403,264 parameters (read from `adapter_model.safetensors`); 4.880 GPU-h measured | 4.90 | 4.90 |
| 2 | 5228013 | A seed 42 audit against SFT-v3: +0.13 pp, every guardrail passes, conforms; merge kept 95.7% | 1.21 | 6.11 |
| 3 | 5228130 | A seed 43 training: 341 steps, 1:12:48; 26 modules, 11,403,264 parameters; 4.836 GPU-h measured | 4.85 | 10.96 |
| 4 | 5228131 | A seed 44 training: 341 steps, 1:12:20; 26 modules, 11,403,264 parameters; 4.803 GPU-h measured | 4.82 | 15.78 |
| 5 | 5228155 | Full-grid margins, GRPO R4 s42 and A s42 (two GPUs of four); gate 99.65% / 99.57% | 2.17 | 17.95 |
| 6 | 5228380 | Patching, sel_sft_v3: GRPO R4 s42 on 3 contrasts, A s42 on implicit combat | 1.53 | 19.48 |
| 7 | 5228526 | Patching of A s42 (explicit combat, atheist) and noise passes (2,048 rows) for GRPO R4 s42 and A s42 | 1.27 | 20.75 |
| 8 | 5228565 | A seed 43 audit against SFT-v3: +0.27 pp, every guardrail passes, conforms | 1.24 | 21.99 |
| 9 | 5228566 | A seed 44 audit against SFT-v3: −0.00 pp, every guardrail passes, conforms | 1.34 | 23.33 |
| 10 | 5228626 | C seed 42 training: 341 steps, 1:13:38; layers 15-20 only, 38 modules (the 26 of arm A plus `in_proj_qkv`, `in_proj_z`, `out_proj` of GDN 16, 17, 18, 20), 15,073,280 parameters (read from `adapter_model.safetensors`); 4.890 GPU-h measured | 4.91 | 28.24 |
| 11 | 5228681 | DPO student training: 341 steps, 1:22:05; all layers, 128 modules, 58,195,968 parameters, trained on DPO_V3_4880 s42 checkpoint-305 merged (read from `adapter_model.safetensors`, `run_config.json`); 5.453 GPU-h measured | 5.47 | 33.71 |
| 12 | 5228682 | GRPO student training: 341 steps, 1:16:19; all layers, 128 modules, 58,195,968 parameters, trained on GRPO + decay R4 s42 merged; 5.068 GPU-h measured | 5.09 | 38.80 |
| 13 | 5229370 | C seed 42 audit against SFT-v3: +0.13 pp, every guardrail passes, conforms; merge kept 94.3% | 1.23 | 40.03 |
| 14 | 5229488 | DPO student audit against SFT-v3: −1.50 pp, every guardrail passes, conforms; merge kept 96.5% | 1.23 | 41.26 |
| 15 | 5229489 | GRPO student audit: failed after the merge, node nid006377 reported "No CUDA GPUs are available" (`logs/xw-eval-arm-5229489.out`); resubmitted, excluding that node | 0.57 | 41.83 |
| 16 | 5229674 | GRPO student audit against SFT-v3 (resubmitted, nid005800): −0.16 pp, every guardrail passes, conforms; merge kept 95.6% (the merged weights the failed job had written were reused) | 1.00 | 42.83 |

## 3. Result

### A. Distillation confined to layers 15-20 (registered call: H0)

Each seed against the all-layer arm of the same seed (`compare-arms`), and against SFT-v3 (the
standard; `reports/comparisons.csv`, `reports/reference_metrics.csv`). Every seed passes every
guardrail and every comparison conforms.

| seed | against all layers, pp [95% CI] | against SFT-v3, pp [95% CI] |
|---|---|---|
| 42 | +0.26 [−0.32, +0.82] | +0.13 [−0.33, +0.59] |
| 43 | +0.74 [+0.01, +1.45] | +0.27 [−0.19, +0.77] |
| 44 | +0.58 [−0.13, +1.24] | −0.00 [−0.45, +0.48] |

- **Fact.** The mean change against all layers is +0.53 pp, above +0.30: by the registered rule,
  **H0, the gain is lost**. No seed is below SFT-v3 by more than 0.01 pp; the mean against SFT-v3
  is +0.13.
- **Fact (patching, seed 42; `reports/interp/distill/pd42/`).** The confined adapter barely moves
  SFT-v3: the attribute effect is 0.88-0.90 of SFT-v3's and the margin scale 0.84, against
  0.39-0.47 and 0.45-0.46 for all-layer distillation. The component profile correlates with
  SFT-v3's at r = 0.997-0.998, and the attribute difference at the decision tokens over depths
  15-20 is unchanged (ratio 1.000).
- **Interpretation, not confirmed.** What all-layer distillation changes, it changes mostly
  outside the window where the attribute reaches the decision. Read with the scale-only control
  (`SCALE_CONTROL.md`), the window is where the attribute arrives, not where the gain is made.
- **Unknown.** Whether training the window's GDN mixers as well (arm C) or a larger learning
  rate changes this; the logs show no underfitting (§0, review).
- **Wording, as registered:** "distillation confined to layers 15-20 at the same learning rate
  loses the gain (mean +0.53 pp against all layers, three seeds)".


### B. The same distillation on the DPO and GRPO models (pilots, one seed each)

**DPO student** (`decision_distill_dpo4880_seed42`, parent DPO_V3_4880 s42 checkpoint-305).
Every guardrail passes against SFT-v3 and both comparisons conform (commit a8be782).

| comparison | pp [95% CI] | hire rate |
|---|---|---|
| against the parent (`compare-arms`, the registered question) | −0.07 [−0.44, +0.33] | 23.16% |
| against SFT-v3 (the standard, context) | −1.50 [−2.42, −0.64] | 23.17% |
| the parent against SFT-v3, for reference | −1.03 [−1.81, −0.28] | 22.80% |

- **Fact.** Against its parent the change is −0.07 pp and its interval holds 0: by the registered
  wording, not distinguishable from the parent. At the matched rate the parent has 0.63% unstable
  sets and the student 0.56% (`reports/operating_point/v2/…dpo4880_seed42--vs--…dpo_v3_4880_step305.json`).
- **Fact.** Against SFT-v3 the student reads −1.50 pp, the lowest of any arm, with 0.59% unstable
  sets (`reports/reference_metrics.csv`). Most of it is the parent's own seed-42 draw, the best of
  DPO's three seeds (−1.03, −0.09, +0.20; mean −0.31). As registered, it is not reported as
  "distilled DPO beats SFT-v3".
- **Fact (decisions).** The student differs from its parent on 276 of 31,050 decisions, from
  SFT-v3 on 475; the parent differs from SFT-v3 on 309.
- **Interpretation, not confirmed.** Distillation moves the DPO model about as much as it moves
  SFT-v3, but the parent is already at 0.6-0.9% unstable sets, so there is little left for the
  attribute-free teacher to remove. The two head-to-head numbers (−0.07 against the parent and
  −0.47 more than the parent against SFT-v3) differ because each SFT-v3 comparison is matched at
  that arm's own hire rate.
- **Unknown.** A second seed of the student; whether the parent's other seeds would gain more.

**GRPO student** (`decision_distill_grpo_r4_seed42`, parent GRPO + lr decay R4 s42). The first
audit failed on a node fault (ledger row 15); the resubmitted one passes every guardrail against
SFT-v3, and both comparisons conform (commit ddc8b39).

| comparison | pp [95% CI] | hire rate |
|---|---|---|
| against the parent (`compare-arms`, the registered question) | +0.02 [−0.77, +0.82] | 21.78% |
| against SFT-v3 (the standard, context) | −0.16 [−1.15, +0.79] | 21.91% |
| the parent against SFT-v3, for reference | −0.73 [−1.64, +0.19] | 20.93% |

- **Fact.** Against its parent the change is +0.02 pp, ≥ 0: by the registered rule, **against**,
  read "not distinguishable from regression to the method mean". The parent is the best of GRPO
  R4's three seeds (−0.73, −0.44, −0.16; mean −0.44).
- **Fact.** Against SFT-v3 the student's TPR change is −1.67 pp against the parent's −4.70, and its
  hire rate moves from 20.93% to 21.91%, toward SFT-v3's 22.75% (`reports/reference_metrics.csv`).
- **Fact (decisions).** The student differs from its parent on 439 of 31,050 decisions, from SFT-v3
  on 521; the parent differs from SFT-v3 on 588.
- **Interpretation, not confirmed.** Distillation pulls the GRPO model back toward SFT-v3 (hire rate,
  TPR, decisions) without changing its stability at a matched rate.

**Reading for B.** Neither student improves on its parent at a matched hire rate (DPO −0.07, GRPO
+0.02; one seed each). G2 is not met. As registered, no windowed student is trained.

### C. The whole pathway window, GDN mixers included (pilot: the gate fails, seeds 43 and 44 not run)

Seed 42, against the all-layer arm of seed 42 (`compare-arms`) and against SFT-v3 (the standard;
`reports/comparisons.csv`, `reports/reference_metrics.csv`). Every guardrail passes and both
comparisons conform (commit a8be782).

| seed | against all layers, pp [95% CI] | against SFT-v3, pp [95% CI] |
|---|---|---|
| 42 | +0.26 [−0.37, +0.89] | +0.13 [−0.27, +0.58] |

- **Fact.** The pilot rule needs ≤ +0.20 against all layers and a change below 0 against SFT-v3.
  Both fail, so seeds 43 and 44 are not run (section 1C).
- **Fact.** The GDN projections were trained: their update norms ‖BA‖·α/r are 0.16-0.29, in
  the range of the window's MLP projections (0.14-0.38), and the merge kept 94.3% of the update
  (`adapter_model.safetensors`, `reports/reference_metrics.csv`). So the result is not a LoRA left
  at zero.
- **Fact (decisions, `outputs/raw/`).** Of 31,050 decisions, arm C differs from SFT-v3 on 144 and
  from arm A seed 42 on 73. Arm A seed 42 differs from SFT-v3 on 171, the all-layer arm of seed 42
  on 472. Its hire rate is 22.84% (SFT-v3 22.75%, arm A 22.88%, all layers 23.41%).
- **Interpretation, not confirmed.** Training the window's mixers as well does not make the
  windowed adapter move SFT-v3's decisions. The same numbers as arm A seed 42 (+0.26, +0.13)
  follow from both staying close to SFT-v3, not from identical models.
- **Wording:** "distillation confined to layers 15-20, GDN mixers included (a quarter of the
  trainable parameters), seed 42: +0.26 pp against all layers; not continued".

## 4. Reading

## 5. Caveats

## 6. Reproducing

```bash
HBM_COST_ARM=patching_distill REPO=$PWD CONFIG=configs/mitigation/distill/decision_distill_l15_20_seed42.yaml \
  scripts/cscs/submit.sh -p normal -t 02:00:00 scripts/cscs/train_distill.sbatch
HBM_COST_ARM=patching_distill REPO=$PWD \
  AUDIT=configs/audit/qwen3.5-9b_en_decision_distill_l15_20_seed42_gh200.yaml \
  AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml \
  scripts/cscs/submit.sh -t 00:30:00 scripts/cscs/evaluate_arm.sbatch
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw OPENBLAS_NUM_THREADS=1 \
  ../venv/bin/python scripts/operating_point_control.py compare-arms \
  --config configs/audit/qwen3.5-9b_en_decision_distill_l15_20_seed42_gh200.yaml \
  --against configs/audit/qwen3.5-9b_en_decision_distill_seed42_gh200.yaml
# seeds 43 and 44 the same; the students:
#   --config configs/audit/qwen3.5-9b_en_decision_distill_dpo4880_seed42_gh200.yaml \
#   --against configs/audit/qwen3.5-9b_en_dpo_v3_4880_step305_gh200.yaml
#   --config configs/audit/qwen3.5-9b_en_decision_distill_grpo_r4_seed42_gh200.yaml \
#   --against configs/audit/qwen3.5-9b_en_grpo_improve_r4_tipped_gh200.yaml
```
