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
| `dpo4880` | `Qwen3.5-9B--en--dpo--adapter--en_only_dpo_v3_4880_step305` (`outputs/dpo/qwen3.5-9b_en_only_dpo_v3_4880/checkpoint-305`) | −1.04 [−1.81, −0.28] | +0.70 |
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
