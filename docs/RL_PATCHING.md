# Activation patching of the best DPO and GRPO arms (registration)

The user asked for this on 2026-10-28 ("Marta推新代码了，跑DPO，GRPO最好模型分析", translated:
"Marta pushed new code; run the analysis on the best DPO and GRPO models"). Branch
`xinyang/interp-distill`. Method and code as in [DISTILL_PATCHING.md](DISTILL_PATCHING.md).

## Marta's new code

Marta's new commit, f5b75d3 (fork `interp_evidence_pipeline`, merged into `origin/main` by
PR #2 as 33b1dd0), adds one file: `src/hiring_bias_mitigation/eval/decision.py`. Her first
commit imported it but did not include it. The port had filled the gap with a re-export of
`eval/margins.py`.

Her version is a standalone implementation. Checked on the Qwen3.5 tokenizer, it gives the same
prefix, the same words and the same token ids for both languages:
- en: hire 33304, reject 45427;
- uk: 196933 / 185152.

So no earlier result changes. The commit is cherry-picked with her authorship (033134d), taking
her file, so that the branch matches main. The interp tests and the distillation tests pass
(41 passed, 1 skipped).

## Setup

**Arms.** These are the two arms whose standard verdict against SFT-v3 is "better"
(`reports/reference_metrics.csv`). Both are fresh LoRAs (r 32 / α 64, the seven projections)
trained on SFT-v3 merged. So each spec stacks [SFT-v3, arm], and SFT-v3 is its reference, as for
distillation.

| spec | run | matched change vs SFT-v3, pp [95% CI] | TPR vs ref |
|---|---|---|---|
| `dpo4880` | `Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305` (`outputs/dpo/qwen3.5-9b_en_only_dpo_v3_4880/checkpoint-305`) | −1.03 [−1.81, −0.28] | +0.70 |
| `grpo525` | `Qwen3.5-9B--en--grpo--adapter--en_only_grpo_v3_all_step525` (`outputs/grpo/qwen3.5-9b_en_only_grpo_v3_all/checkpoint-525`) | −1.02 [−1.99, −0.12] | −9.30 |
| *context:* distillation s43 | (DISTILL_PATCHING.md) | −0.59 [−1.24, −0.03] | +1.29 |

**Question.** SFT (Marta) and distillation (DISTILL_PATCHING.md) both keep the L15 pathway and
turn it down. Distillation also compresses every margin (r_scale 0.46, ρ 0.88–0.99). The DPO and
GRPO arms lower instability about as much or more. Do they do it by the same compression, by
cutting the pathway, or in another way?

**What points to another way, read before any pass** (`reference_metrics.csv`, op-margin-v2,
nats on each model's own scale):

| | within-set margin SD | paired attribute gap | unstable within-group % |
|---|---|---|---|
| SFT-v3 | 0.216 | 0.232 | 1.85 |
| distillation s43 | 0.121 | 0.094 | 1.52 |
| DPO 4880 | 0.361 | 0.356 | 0.89 |
| GRPO 525 | 0.289 | 0.276 | 1.04 |

Distillation shrinks both. DPO and GRPO enlarge both, yet flip fewer sets. That fits margins
pulled away from the threshold faster than the attribute effect grows.

**Measures.** These are the three from DISTILL_PATCHING.md, unchanged. Each arm is compared with
SFT-v3 on the full grid (31,050 prompts) and on the same three contrasts, with the same 64
pairs per contrast: selected on SFT-v3's own effect (|Δm| ≥ 0.5), seed 42, identical to the
SFT-v3 patching runs already on disk.
- (a) r_scale, r_attr, ρ = r_attr / r_scale.
  - Registered raw, with the noise-corrected value beside it.
  - The noise floor comes from a second pass on 2,048 rows at chunk 256, as before.
- (b) Components kept per pair, arm over SFT-v3.
  - Hub (attention mixers L11 and L15) against readout (MLP31).
  - The profile correlation of the kept components with SFT-v3's.
  - The top AtP heads.
- (c) Last-position residual attribute difference, arm over SFT-v3, by depth.
  - The mean over depths 15–20 and the final depth.

**Predictions** (each arm separately; "on ≥ 2 of 3 contrasts" wherever per-contrast).
- *Scale.* Sharpened if r_scale > 1.10, compressed if < 0.90, otherwise unchanged.
  - Predicted: sharpened for both, from the table above.
- *Attribute relative to scale.*
  - ρ < 0.85: attribute-specific. Predicted for both, since they flip fewer sets with larger
    margins.
  - 0.85 ≤ ρ ≤ 1.15: uniform.
  - ρ > 1.15: amplified.
- *Where.*
  - Same pathway: component profile r ≥ 0.80 against SFT-v3 and L15.H11 among the top 3 AtP
    heads in each contrast.
  - Pathway cut at the hub: the hub keeps ≥ 0.15 less than the readout and (c) at depths 15–20
    is < 0.85.
  - Readout only: hub and readout within 0.15 of each other and (c) at depths 15–20 in
    [0.85, 1.15].
  - Anything else is mixed and reported as such.
- What would count against the sharpening reading: r_scale ≤ 1.10 for an arm, or ρ ≥ 0.85.

**Gate before any patching.** The full-grid HF margins must agree with each arm's own audit
decisions on ≥ 97% of rows. The earlier specs reached 99.2–99.7%. If an arm falls below, stop
and diagnose; that would mean the adapter chain is wrong.

**Splits.** No training and no model selection. This reads two trained models on the benchmark
grid, and nothing feeds back into a checkpoint or a hyperparameter.

**Cost.** From the earlier records:
- 0.53 GPU-h for a full margin pass;
- 0.07 for a noise pass;
- 0.33 per patching run.

Two arms ≈ 2 × (0.53 + 0.07 + 3 × 0.33) ≈ 3.2 GPU-h, with a cap of 5. It is recorded under the
arm `rl_patching`. It is a measurement, not a training round, so it draws on no training
budget.

**Jobs, one at a time.**
1. Full margin passes, `--resid --batch-size 16`, both arms.
2. `04_manifest.py`, then the agreement gate.
3. Four patching runs.
4. Two patching runs plus both noise passes.

## Result (2026-10-29)

**Jobs and cost.** Margin passes 5190441, patching 5190857 (DPO × 3, GRPO implicit combat) and
5198222 (GRPO × 2, noise passes). 3.18 GPU-h against the estimate of 3.2 and the cap of 5
(`$XW/cost/records/*_{5190441,5190857,5198222}_*.json`, arm `rl_patching`). Code at
296982b; the summary columns at 9a843df.

**Gate: passed.** The full-grid HF margins agree with each arm's own audit decisions on 99.79%
(DPO) and 99.77% (GRPO) of 31,050 rows. The noise passes agree on 99.80% and 99.90% of 2,048
(`reports/interp/margins_summary-{dpo4880,grpo525}-grid-en*.json`).

Files: `reports/interp/distill/{dpo4880,grpo525}/{sensitivity,components,component_summary,
heads,residual,noise_floor}.csv` and the per-pair patching under
`reports/interp/patching/{dpo4880,grpo525}-grid-*-n64-s42-sel_sft_v3/`.

**(a) Sensitivity**, against SFT-v3 on all 450 pairs per contrast.

| arm, contrast | r_attr | r_scale | ρ (registered) [95% CI] | ρ_rms | ρ noise-corrected [95% CI] |
|---|---|---|---|---|---|
| DPO, implicit combat | 1.331 | 2.685 | **0.496** [0.35, 0.66] | 1.306 | 1.280 [0.83, 1.74] |
| DPO, explicit combat | 1.200 | 2.685 | **0.447** [0.34, 0.58] | 0.887 | 0.787 [0.50, 1.14] |
| DPO, atheist implicit | 2.107 | 2.685 | **0.785** [0.55, 1.05] | 1.967 | 1.962 [1.36, 2.56] |
| GRPO, implicit combat | 0.975 | 2.203 | **0.443** [0.37, 0.53] | 0.741 | 0.735 [0.53, 0.94] |
| GRPO, explicit combat | 1.216 | 2.203 | **0.552** [0.46, 0.65] | 0.915 | 0.912 [0.70, 1.08] |
| GRPO, atheist implicit | 1.378 | 2.203 | **0.625** [0.50, 0.77] | 1.051 | 1.054 [0.75, 1.37] |

Per-row margin noise, as the SD between the two passes: SFT-v3 0.082 logits, DPO 0.417, GRPO
0.168. It scales with the margins.

**(b) Components**, real patching on the 64 pairs per contrast selected on SFT-v3's own
|Δm| ≥ 0.5, logits per pair, arm over SFT-v3 (`component_summary.csv`).

| arm, contrast | total kept | hub (L11+L15 attn) | readout (MLP31) | hub − readout | profile r | layers ≤11 / 12–20 / ≥21 | L15.H11 rank (arm's own AtP) | arm's top 3 heads |
|---|---|---|---|---|---|---|---|---|
| DPO, implicit combat | 1.81 | 1.84 | 1.38 | +0.46 | 0.65 | 1.35 / 1.95 / 2.48 | 6 | L23.H1 L17.H0 L19.H11 |
| DPO, explicit combat | 1.30 | 0.85 | 1.11 | −0.27 | 0.68 | 1.10 / 1.21 / 1.44 | 3 | L7.H3 L7.H4 L15.H11 |
| DPO, atheist implicit | 2.29 | 2.68 | 1.82 | +0.86 | 0.89 | 2.73 / 2.69 / 2.86 | 2 | L19.H11 L15.H11 L7.H8 |
| GRPO, implicit combat | 0.99 | 1.13 | 0.96 | +0.17 | 0.92 | 1.01 / 1.01 / 1.06 | 1 | L15.H11 L17.H0 L19.H11 |
| GRPO, explicit combat | 1.39 | 1.19 | 1.48 | −0.29 | 0.91 | 1.27 / 1.29 / 1.62 | 1 | L15.H11 L17.H0 L19.H11 |
| GRPO, atheist implicit | 1.43 | 1.48 | 1.62 | −0.15 | 0.93 | 1.36 / 1.48 / 1.76 | 2 | L7.H8 L15.H11 L19.H11 |
| *context:* distillation s43 | 0.35–0.41 | 0.39–0.44 | 0.25–0.31 | +0.10 to +0.16 | 0.91–0.98 | | 1 / 1 / 1 | |

On the pairs where SFT-v3 was biased, both arms carry the attribute effect *more* strongly in
absolute logits: total kept 1.30–2.29 for DPO, 0.99–1.43 for GRPO. Divided by r_scale that is
0.48–0.85 and 0.45–0.65.

**(c) Residual**, the last-position attribute difference, arm over SFT-v3.
- Mean over depths 15–20: DPO 0.94 / 0.97 / 1.00; GRPO 0.98 / 1.00 / 1.00 (implicit combat,
  explicit combat, atheist).
- At the final depth: DPO 1.00 / 0.97 / 1.17; GRPO 0.86 / 0.90 / 0.94.
- The noise-corrected values differ by at most 0.06.

**Against the registration.**
- *Scale.* Sharpened for both, as predicted (r_scale 2.69 and 2.20 > 1.10).
- *Attribute relative to scale.* On the registered statistic, ρ < 0.85 on all three contrasts
  for both arms: attribute-specific, as predicted. The "count against" condition (ρ ≥ 0.85) is
  not met.
  - The noise-corrected value registered beside it does not agree. It uses root-mean-square
    |Δm|, which weights the tail: DPO 1.28 / 0.79 / 1.96 (amplified on two contrasts), GRPO
    0.74 / 0.91 / 1.05.
  - The registered verdict therefore holds for the typical pair, not for the effect's tail.
    Per-pair quantiles of |Δm|, arm over SFT-v3 (`sensitivity.csv`, `r_attr_median` and
    `r_attr_p99`, added after the result):

    | arm | median | 99th percentile |
    |---|---|---|
    | DPO | 0.29 / 0.36 / 0.35 | 6.26 / 4.46 / 10.69 |
    | GRPO | 0.44 / 0.53 / 0.52 | 1.95 / 2.80 / 3.28 |
    | distillation s43 | 0.45 / 0.51 / 0.52 | 0.40 / 0.41 / 0.43 |
- *Where.*
  - GRPO: **same pathway** (profile r 0.91–0.93 ≥ 0.80; L15.H11 in its own top 3 on every
    contrast). Neither "cut at the hub" nor "readout only" holds on all contrasts: hub − readout
    is +0.17 / −0.29 / −0.15, and the residual at 15–20 is about 1.
  - DPO: **not the same pathway** by the registered rule. Profile r is below 0.80 on both
    combat contrasts, and L15.H11 ranks 6th on implicit combat, where the late head L23.H1
    leads. The amplification grows with depth on both combat contrasts. Mixed.

## Reading

- **Fact.**
  - Both arms widen every margin about 2–2.7× and keep the attribute signal arriving at layers
    15–20 unchanged.
  - GRPO carries the attribute effect through SFT-v3's pathway, a little more strongly in
    absolute terms.
  - DPO spreads it differently, more in the late layers.
  - For the typical pair the attribute effect shrinks; for a few pairs it grows a lot.
- **The widening is not the explanation.** By SCALE_CONTROL.md P1, a rescaling of SFT-v3's
  margins, by any a, leaves the matched statistic at exactly 0. Both arms' gains survive
  unchanged against SFT-v3 widened by their own a. So "DPO and GRPO stabilize decisions by
  widening margins" is not supported. It stays an unconfirmed reading, and the margin
  measurements here do not test it.
- **What the post-hoc hybrids suggest, not confirmed** (SCALE_CONTROL.md):
  - The arms' set centers, i.e. where each pair lands relative to the cut beyond the rescaling,
    would alone give about twice their gain (DPO −1.99, GRPO −1.64).
  - Their own within-set deviations cost it back (+3.52, +1.19).
  - This is consistent with the heavy-tailed attribute effect above.
- **Unknown.**
  - Why the centers move.
  - Whether DPO's late heads (L23.H1) carry the large-effect pairs.
  - One seed per arm.

## Caveats

- **The noise correction over-corrects for these arms.** Its noise floor compares two passes
  with different batch neighbours. Within one pass the two prompts of a pair are scored
  together and share most of that noise. DPO's median |Δm| (about 0.1) is below its
  between-pass row noise (0.42).
- **Patched pairs are selected on SFT-v3's effect**, not on the arms' large-effect pairs. The
  arms' tail pairs are mostly outside them, so (b) describes where the effect flows on
  SFT-v3's biased pairs, not on DPO's.
- **Heads are AtP estimates.** The real-patching check is in each run's `verify.csv` and is not
  summarised here.
