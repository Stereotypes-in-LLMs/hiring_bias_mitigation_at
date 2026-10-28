# Layer-restricted decision distillation (LoRA on layers 0–20)

Asked by the user on 2026-10-28 ("可以，试一下"), following the layer-masked check in
[DISTILL_PATCHING.md](DISTILL_PATCHING.md). Branch `xinyang/interp-distill`. The cost is
charged to the decision-distillation budget: 23.57 of 50 GPU-h spent before this round
([DECISION_DISTILL.md](DECISION_DISTILL.md) plus the patching).

## 1. Registration (2026-10-28, before any run)

**Where the idea comes from.** The masked check (job 5185597) switched the trained seed-43
adapter off outside layers 0–20 or outside layers 21–31. The comparison is against SFT-v3 on
the 450-pair grid (`reports/interp/distill/distill43/mask_shares.csv`).
- The global margin scale, r_scale (full adapter 0.464), split about evenly: 0.49 of the
  log-shrink came from layers 0–20 and 0.54 from layers 21–31.
- The attribute effect, r_attr (full 0.41–0.46), came more from layers 0–20 (0.57–0.60)
  than from layers 21–31 (0.36–0.38).
- On its own, the early half shrinks the attribute effect more than the scale, ρ 0.85–0.94.
  The late half does the opposite, ρ 1.09–1.13.

Those halves were never trained alone. Each was trained with the other present.

**Question.** Does decision distillation with the LoRA on layers 0–20 only still beat SFT-v3,
and as much as the full adapter? Does it keep more of the global margin scale, as the masked
check suggests?

**Hypothesis H1.** The part of distillation that changes decisions is the attribute-specific
attenuation from layers 0–20. A rescaling of every margin changes few decisions at matched
hire rate. So an adapter trained on layers 0–20 alone:
- keeps the decision-level gain;
- keeps more of the margin scale;
- shows a lower ρ than the full adapter.

**Alternative H0.** The masked shares do not predict what training does. With only layers 0–20
trainable, the optimizer puts the global compression there as well, giving r_scale near 0.46
and ρ near 1. Or it loses the gain, because the late layers were needed.

**Change (one variable).** `lora_layer_range: [0, 20]`. This keeps 83 of the 128 LoRA modules:
the 21 MLPs and 5 full-attention layers in 0–20. The module counts match the masked check.
- Config: `configs/mitigation/distill/decision_distill_l0_20_seed43.yaml`.
- Everything else matches `decision_distill_seed43`:
  - data `configs/data/en_only_distill_v3_all.yaml`;
  - lr 3e-5, wsd decay over the last 89 steps;
  - 340 steps;
  - r 32 / α 64;
  - seed 43, chosen to match the adapter the masked check was run on.
- The learning rate is not retuned for the smaller adapter. Retuning would be a second variable.

**Controls, matched scope by op-margin-v2.**
- The standard: against SFT-v3, `Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`.
- Same seed, all layers: compare-arms against `Qwen3.5-9B--en--sft--adapter--en_only_decision_distill_seed43`.
  - That arm's matched change against SFT-v3 was −0.59 [−1.24, −0.03], with TPR +1.29.

**Model selection.** The final adapter (step 340), declared now. Nothing is selected on the
benchmark. The grid margin pass in step 4 below is a measurement after the audit, not a selection.

**Expected observations and readings.**

| | decisions: gain kept | decisions: gain lost |
|---|---|---|
| **scale kept** (r_scale ≥ 0.60, ρ below the full adapter's on ≥ 2 of 3 contrasts) | H1 holds. The late layers are not needed for the gain | the scale compression was doing work after all |
| **scale not kept** (r_scale < 0.55) | the gain survives, but the masked shares do not predict training (H0, first part) | H0 |

- *Gain kept* means both of these:
  - the standard's verdict passes every guardrail with matched change < 0 against SFT-v3;
  - compare-arms against distillation s43 gives a point change ≤ +0.20 pp.
- *Gain lost* means any one of these:
  - change ≥ 0 against SFT-v3;
  - a failed guardrail;
  - a point change > +0.30 pp against distillation s43.
- Between +0.20 and +0.30 pp is undecided.
- One seed cannot separate two arms that sit 0.2 pp apart. The interval against s43 is
  reported, and any claim that the arm is *better* than the full adapter needs further seeds.

**Stop conditions.**
- The smoke must record `lora_modules` = 83 in `run_config.json` and finite attributed and
  anchor KLs. Otherwise stop.
- `memory_floor_gb: 12`.
- A cap of 8 GPU-h for the round.
- One seed. Further seeds only on the user's decision.

**Steps and cost.**

| # | Step | Estimate |
|---|---|---|
| 1 | smoke, six steps on `debug` | 0.3 GPU-h |
| 2 | training, seed 43, 4 GPUs | 5.2 GPU-h (seed 43 full took 5.26) |
| 3 | evaluate_arm against SFT-v3, plus compare-arms against s43 | 0.5 |
| 4 | interp margin pass on the grid with `--resid`, plus `20_distill_compare.py` against `sft_v3` | 0.6 |
| | total | ≈ 6.6, cap 8 |
