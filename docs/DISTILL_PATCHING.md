# Activation patching of decision distillation: pathway or readout? (registration)

Branch `xinyang/interp-distill`, from `xinyang/decision-distill` with Marta Samoilenko's
patching pipeline (commit 280b8f8 on the fork, cherry-picked as ccaa45d) on top. The user asked
for it on 2026-10-28 ("试试patching的方法，看下有什么结论给我，看下Marta的commit", translated:
"try the patching method, tell me what it concludes, look at Marta's commit").

## Setup

**Question.** Decision distillation (`DECISION_DISTILL.md`) lowers matched instability against
SFT-v3 in all three seeds. Section 5b found the attribute-free margins compressed as a whole
(median |m| 12.9 → 6.0–6.6), and read the halved within-set spread as that global compression,
not invariance. Does distillation remove attribute sensitivity only by turning the readout down
everywhere (H0), or does it also weaken the pathway the attribute travels on (H1)? Marta's
report (`reports/interp/latex/activation_patching_report.pdf`) locates that pathway in the base
model: layers 15–20, hub at full-attention layer 15 (head L15.H11), readout through MLP31. It
found her SFT (the Hub adapter, not SFT-v3) keeps the pathway and turns it down.

**What was ported, and nothing else changed in her method.**
- `eval/decision.py`, which her code imports and which was not in her commit: it re-exports
  `PREFIX`, `WORDS` and `decision_token_ids` from `eval/margins.py`, so there is one definition.
- Specs `sft_v3`, `distill42`, `distill43` in `configs/interp/models.yaml`. A distillation spec
  stacks [SFT-v3, distillation], since the distillation LoRA trained on SFT-v3 merged (the
  identity her `[sft, dpo]` spec relies on).
- `10_patch.py --select-on SPEC`: pairs selected on the reference spec's own effect. Her
  selection uses the base model; for this question the reference is SFT-v3, whose effect is a
  quarter of the base's, so base-selected pairs would include many where SFT-v3 has nothing
  left to remove. Never selected on the distilled model. A non-base selection is part of the
  run name (`-sel_sft_v3`).
- `scripts/cscs/interp.sbatch`: up to four tasks per GH200 node. GDN kernels from `fla` 0.6.0,
  installed into `$XW/pylib-fla` (on `PYTHONPATH` for these jobs only), as in her runs.
- Same torch 2.13.0, transformers 5.17.0, peft 0.21.2 as her `env.json`. The grid segments
  identically: `reports/interp/segments_summary-grid.json` is unchanged byte for byte (0 hard
  violations, 2,700 sets, prompt length p50 719, max 1,851).

**Control and scope.** SFT-v3 (`sft_v3`), the distillation's start and the standard's reference
arm. Primary distilled model: seed 43 (`distill43`, the seed whose interval is below 0); seed 42
replicates the margin-level measures. Grid = the 31,050 benchmark prompts. Patching contrasts are
her three, fixed in advance by the baseline audit: `mil_combat_civ_impl`, `mil_combat_civ_expl`,
`rel_atheist_christian_impl`; 64 aligned pairs each (seed 42) with |Δm_SFT-v3| ≥ 0.5 logits,
identical for both specs. Base margins are run too, because her manifest requires them.

**Measures, fixed now.**
- (a) *Normalised sensitivity*, from the fp32 margins on the full grid: ρ = r_attr / r_scale,
  with r_attr = mean |Δm| of distill over SFT-v3 on every aligned pair of a contrast, and
  r_scale = median |m| on the attribute-free prompts, distill over SFT-v3. ρ = 1 means the
  attribute effect shrank exactly as much as the margins did.
- (b) *Where it shrank*, from real patching on the identical pairs (suffix positions): each
  component's contribution in logits per pair, distill over SFT-v3 ("kept"). Hub = attention
  layers 11 and 15; readout = MLP31; heads by AtP, L15.H11 first.
- (c) *Inside the residual*, from the stored last-position residuals (`--resid`): per layer,
  ‖r_L(A) − r_L(B)‖ / ‖r_L(B)‖ averaged over the pairs, distill over SFT-v3. This does not go
  through the readout, so a pure readout change leaves layers 15–20 at 1.

**Predictions.**
- H0, readout only: ρ in [0.85, 1.15] on at least two of three contrasts; hub and readout kept
  shares within 0.15 of each other; (c) at layers 15–20 within [0.85, 1.15].
- H1, the pathway too: ρ < 0.85 on at least two of three contrasts; the hub keeps at least 0.15
  less than the readout; (c) at layers 15–20 below 0.85.
- Anything else is mixed and is reported as such.

**Stop conditions.** A 64-row margin smoke on `debug` for all four specs first: decision
agreement of the HF margins with each spec's own audit ≥ 97% (hers: 99.0–99.2%), else stop and
diagnose before any full pass. A cap of 16 GPU-h for this round, from decision distillation's
unspent 32.7 of 50.

**Splits.** No training and no model selection: this reads three trained models on the
benchmark grid. Nothing here feeds back into a checkpoint or a hyperparameter.

**Cost, before the smoke.** Her pass: 1.4 GPU-h per spec on an RTX 6000 Ada. Four specs ≈ 6,
six patching runs ≈ 6–8 (to be re-estimated from the smoke), smoke ≈ 0.5.
