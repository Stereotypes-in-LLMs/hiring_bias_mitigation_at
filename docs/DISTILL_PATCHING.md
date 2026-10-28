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

**Smoke (job 5184624, 2026-10-28, 3.5 min on debug).** 64 grid rows per spec, fla kernels.
Decision agreement with each spec's own audit: base 100%, distill43 100%, distill42 100%,
sft_v3 95.3%, below the 97% line. The three sft_v3 disagreements are ties: HF fp32 margins
−0.06, −0.19 and +0.06 logits, against vLLM's +0.25, +0.125 and −0.125 in the op-margin-v2 pass;
over the 64 rows, HF fp32 against vLLM correlates at r = 0.99976 with a median |difference| of
0.087 logits (bf16 spacing is 0.125). `reports/interp/margins_summary-*-grid-en-limit64.json`.

**Amendment, before any full pass or result (2026-10-28).** On 64 rows one tied set decides
the agreement share, so the stop condition moves to the full grid, where `02_margins.py` reports
it on 31,050 rows: agreement ≥ 97% there, else stop. The smoke's purpose, whether the HF spec is
the evaluated model, is met by the margin correlation above.

**Second amendment, after measures (a) and (c) were computed (2026-10-28), so post hoc.** The
same prompts scored in two passes with different batch neighbours differ by a mean 0.087 logits
(64 rows, SFT-v3, both with fla dispatch on; dispatch off moves them by the same amount), i.e.
each fp32 margin carries a batch-dependent bf16 noise of about 0.08 logits. Additive noise
inflates mean |Δm| more for the spec with the smaller effect, which biases ρ in (a) towards 1,
and the residual difference in (c) likewise. Added: a noise-floor pass (each spec's first
2,048 grid rows rescored with different batch neighbours, `--chunk 256`) and noise-corrected
versions of (a) and (c), reported next to the registered values. The registered values are not
replaced; the correction was decided with them in view.

**Patching job 5184970 died** in the AtP backward: fla 0.6 sends one GDN backward kernel to
tilelang, which needs nvcc. `interp.sbatch` now sets `FLA_DISABLE_BACKEND_DISPATCH=1`; a check
(job 5185032) ran 4 pairs through all 194 patches, AtP and the head verification.

## Result (2026-10-28)

**One sentence.** Fact: distillation keeps SFT-v3's attribute pathway exactly where it was (same
hub, same top head, same depth) and turns every part of it down to about a third, a little more
at the end of the network than at the hub. Interpretation: the attribute effect shrinks because
the whole decision readout is compressed, not because the pathway was cut. Unknown: whether
this is what makes the stability gain, or only how the KL loss happens to reach its target.

Jobs: margins 5184648 (31,050 rows × 4 specs), noise floor 5185093, patching 5185108 and 5185258
(64 identical pairs per contrast, selected on SFT-v3's |Δm| ≥ 0.5; seed 42). Code 8ea29b4.
Per-pair and per-component files: `reports/interp/patching/<spec>-grid-<contrast>-n64-s42-sel_sft_v3/`,
`reports/interp/distill/{distill43,distill42}/{sensitivity,components,heads,residual,noise_floor}.csv`.
Key curve: `reports/interp/distill/distill43/residual.svg`.

**Margin pass reproduces the audits** (amended check): HF fp32 decisions agree with each spec's
own audit on 99.23% (base), 99.67% (SFT-v3), 99.60% (distill43), 99.62% (distill42) of 31,050
rows (`reports/interp/margins_summary-*-grid-en.json`).

**(a) Sensitivity.** Mean |Δm| on all 450 pairs, SFT-v3 → distill43: 0.499 → 0.204 (implicit
combat), 0.345 → 0.157 (explicit combat), 0.409 → 0.189 (atheist vs Christian, implicit); all
attribute-free margins shrink by r_scale 0.464 (distill42: 0.446).

| ρ, distill / SFT-v3 | registered | 95% CI | noise-corrected (post hoc) | 95% CI |
|---|---|---|---|---|
| implicit combat, s43 | 0.884 | 0.79–0.96 | 0.849 | 0.75–0.94 |
| explicit combat, s43 | 0.982 | 0.88–1.05 | 0.920 | 0.82–1.00 |
| atheist, implicit, s43 | 0.994 | 0.89–1.05 | 0.934 | 0.84–0.99 |
| implicit combat, s42 | 0.873 | 0.78–0.95 | 0.854 | 0.76–0.95 |
| explicit combat, s42 | 1.043 | 0.95–1.12 | 0.959 | 0.87–1.04 |
| atheist, implicit, s42 | 1.030 | 0.93–1.09 | 0.949 | 0.87–1.00 |

Margin noise per row: SFT-v3 0.082 logits, distill43 0.046, distill42 0.043 (it scales with the
margins), so the correction is small; most of the change from the registered column is the
switch from mean |Δm| to root-mean-square.

**(b) Components** (real patching, suffix positions, logits per pair, distill / SFT-v3):

| run | hub (attn L11 + L15) kept | readout (MLP31) kept | hub − readout | L15.H11 (AtP) kept | profile r | kept, layers ≤11 / 12–20 / ≥21 |
|---|---|---|---|---|---|---|
| implicit combat, s43 | 0.390 | 0.249 | +0.141 | 0.335 | 0.908 | 0.46 / 0.35 / 0.30 |
| explicit combat, s43 | 0.439 | 0.281 | +0.157 | 0.408 | 0.957 | 0.41 / 0.38 / 0.32 |
| atheist, implicit, s43 | 0.413 | 0.309 | +0.104 | 0.457 | 0.979 | 0.40 / 0.37 / 0.33 |
| implicit combat, s42 | 0.357 | 0.219 | +0.138 | 0.326 | 0.954 | 0.34 / 0.34 / 0.26 |
| atheist, implicit, s42 | 0.432 | 0.297 | +0.135 | — | 0.973 | 0.42 / 0.39 / 0.33 |

Under SFT-v3 the biggest carrier is still the layer-15 attention mixer (0.49 and 0.58 logits per
pair on the two implicit contrasts) and the top head is still L15.H11 (0.29, 0.23, 0.18), the
hub Marta's report found in the base model. The residual sweep places the arrival at the same
depth in every spec: share reaching the decision tokens at layer 18 is 0.48 / 0.51 / 0.49
(SFT-v3 / distill43 / distill42, implicit combat), at layer 20 0.85 / 0.88 / 0.88.

**(c) Residual**, attribute difference at the last position, distill43 / SFT-v3: about 1.0 to
layer 8, then falling slowly; mean over depths 15–20 0.873 / 0.841 / 0.886 (registered;
noise-corrected 0.847 / 0.827 / 0.867), at the final pre-norm residual 0.71 / 0.67 / 0.71.
distill42: 0.888 / 0.857 / 0.891, final 0.72 / 0.67 / 0.70.

**Against the registration.** H1 (the pathway is cut) fails on every measure: ρ < 0.85 on at
most one contrast, the hub keeps *more* than the readout in all five runs, and the residual at
15–20 is below 0.85 on one of six. H0 (readout only) holds on (a), on (b) in four of five runs
(explicit combat at +0.157, just past 0.15), and on (c) in five of six (implicit combat s43
0.841). By the rule fixed in advance this is "mixed", and it lies next to H0.

## Reading

- **Supported.** Distillation does not reroute or cut the attribute pathway. It turns the whole
  pathway down by a near-uniform factor, slightly more with depth, most at the final MLP and
  the readout. At the output the attribute effect shrinks about as much as every margin does
  (ρ 0.85–1.04). This is the mechanistic version of section 5b's global scale compression.
- **Also supported, and small.** Inside the residual the attribute difference falls to about
  0.85 of SFT-v3's by layers 15–20, before the readout, in both seeds: a modest attenuation
  beyond pure scaling, strongest on implicit combat, the contrast with ρ consistently lowest.
- **Contrast with Marta's SFT** (base → the Hub SFT adapter, a different comparison): her SFT
  cut most at the movers (attention L11 and L15 kept 15% and 19%, MLP31 28%). Distillation on
  top of SFT-v3 does the reverse: the hub keeps 0.36–0.44, the readout 0.22–0.31.
- **Not shown.** That the compression causes the stability gain, rather than the borderline
  pairs moving (section 5b); whether a loss that keeps the margin scale would still remove the
  attribute effect.

## Caveats

- Pairs are selected on SFT-v3's own |Δm|, so its effect on them is inflated by its noise
  (regression to the mean): "total effect kept" on the patched pairs (0.31–0.41) is below
  r_attr on all pairs (0.39–0.47). Ratios between components are unaffected.
- Three contrasts, 64 pairs each, one selection seed; explicit combat has 92 eligible pairs.
- The noise correction was added after seeing (a) and (c); the registered values are kept.
- Head numbers are AtP estimates; the real-patching verification is in each run's `verify.csv`
  and was not summarised here.
- Margins ran with fla dispatch on, patching with it off. The only dispatched kernel is a
  backward one (tilelang `chunk_bwd_dqkwg`), and a 128-row rerun with dispatch off differs from
  the full pass by the same 0.09 logits as a rerun with other batch neighbours (job 5185032).

## Cost

Smoke 0.15, margins 1.91, failed patching 0.23, check 0.09, noise floor 0.25, patching 1.23 +
1.31 = **5.17 GPU-h** (`reports/cost/steps.csv`, jobs 5184624–5185258), within the 16 cap.
Decision distillation's budget: 17.32 + 5.17 = 22.49 of 50.

## Reproducing

```bash
export REPO=$PWD HBM_COST_ARM=decision_distill
$PY scripts/interp/03_segments.py --dataset grid                                  # CPU
TASKS="margins:base margins:sft_v3 margins:distill43 margins:distill42" \
  MARGIN_ARGS="--resid --batch-size 16" scripts/cscs/submit.sh -t 04:00:00 scripts/cscs/interp.sbatch
TASKS="margins:base margins:sft_v3 margins:distill43 margins:distill42" \
  MARGIN_ARGS="--limit 2048 --chunk 256 --batch-size 16 --resid" \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/interp.sbatch           # noise floor
$PY scripts/interp/04_manifest.py --dataset grid                                  # CPU
TASKS="patch:sft_v3:mil_combat_civ_impl:sft_v3 patch:distill43:mil_combat_civ_impl:sft_v3 ..." \
  scripts/cscs/submit.sh -t 03:00:00 scripts/cscs/interp.sbatch
$PY scripts/interp/20_distill_compare.py --spec distill43 --noise grid-en-limit2048 \
  --contrast mil_combat_civ_impl --contrast mil_combat_civ_expl --contrast rel_atheist_christian_impl
```

## Layer-masked check (registration, 2026-10-28)

Asked by the user ("Run masked check") after the result above.

- **Question.** Where does distillation's compression act: in layers 0–20 (the hub and the
  arrival) or 21–31 (the late layers and the readout)?
- **Change.** Two specs of the seed-43 adapter with its LoRA scaling set to 0 outside a layer
  range (`spec.mask_adapter_layers`; `adapter_layers` in `configs/interp/models.yaml`):
  `distill43_l0_20` and `distill43_l21_31`, each stacked on SFT-v3 as `distill43` is. These are
  not trained models: each half was trained with the other present.
- **Measure.** Full-grid margins with `--resid` (as for the other specs), then
  `20_distill_compare.py` against SFT-v3. Share of the full adapter's shrink a half reproduces,
  on a log scale: s = log r_half / log r_full, for r_scale (all attribute-free margins) and for
  r_attr on each of the three contrasts.
- **Predictions.** Late: s(l21_31) ≥ 0.7 and s(l0_20) ≤ 0.3, for r_scale and for r_attr on all
  three contrasts. Early: the reverse. Anything else: distributed, reported with s(l0_20) +
  s(l21_31) (1 means the halves add on a log scale).
- **Validity check, before reading s.** A residual at depth d depends only on layers < d, so at
  depths ≤ 21 `distill43_l21_31` must match SFT-v3 (residual ratio 1) and `distill43_l0_20` must
  match `distill43`, up to the batch noise floor; the logs must show 83 LoRA modules kept and 45
  off for 0–20, 45 kept and 83 off for 21–31 (21 × 3 MLP + 5 × 4 attention, 11 × 3 + 3 × 4).
- **Cost.** Two margin passes in one job, ≈ 1.2 GPU-h, from decision distillation's budget
  (22.49 of 50 used).

### Layer-masked result (job 5185597, 1.08 GPU-h)

Validity, as registered: 83 LoRA modules kept / 45 off (0–20) and 45 / 83 (21–31) in the job log;
at residual depths 1–21 `distill43_l0_20` equals `distill43` (ratio 1.000 at every depth) and
`distill43_l21_31` equals SFT-v3 (1.000). Files: `reports/interp/distill/distill43/mask_shares.csv`,
`mask_validity.csv`, and each half's `sensitivity.csv` / `residual.csv`.

| against SFT-v3 | full adapter | layers 0–20 only | layers 21–31 only | s, 0–20 | s, 21–31 | sum |
|---|---|---|---|---|---|---|
| r_scale (all attribute-free margins) | 0.464 | 0.686 | 0.662 | 0.49 | 0.54 | 1.03 |
| r_attr, implicit combat | 0.410 | 0.585 | 0.724 | 0.60 | 0.36 | 0.96 |
| r_attr, explicit combat | 0.455 | 0.635 | 0.740 | 0.58 | 0.38 | 0.96 |
| r_attr, atheist, implicit | 0.461 | 0.642 | 0.747 | 0.57 | 0.38 | 0.95 |
| ρ (registered statistic), three contrasts | 0.88 / 0.98 / 0.99 | 0.85 / 0.93 / 0.94 | 1.09 / 1.12 / 1.13 | | | |

- **Against the registration.** Neither half reaches s ≥ 0.7 on either statistic: the check
  reads *distributed*. On a log scale the halves add (sums 0.95–1.03).
- **Fact.** The global compression splits about evenly between the halves (0.49 / 0.54). The
  attribute shrink does not: 0.57–0.60 of it comes from layers 0–20, 0.36–0.38 from 21–31. The
  early half alone shrinks the attribute effect more than the margins (ρ 0.85–0.94); the late
  half alone shrinks the margins more than the attribute effect (ρ 1.09–1.13).
- **Reading, not confirmed.** Two parts. Layers 21–31 compress the readout as a whole, and
  are not attribute-specific. Layers 0–20, where the hub and the arrival are, do the modest
  attribute-specific part, the 0.85 seen in the residual at depths 15–20. This revises the
  earlier reading "most at the readout": per component the late layers lose most, but over the
  full adapter the early half removes more of the attribute effect.
- **Caveat.** The halves are not trained models; each was trained with the other present, so
  "share" is the effect of switching one half off, not what training either half alone would
  give.
