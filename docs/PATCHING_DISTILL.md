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

## 3. Result

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
