# Three-seed mechanism follow-up

All six added jobs passed their completion and artifact checks.

| Method | Real component Pearson r | p50 ratio | p99 ratio | p50 down, p99 up |
|---|---:|---:|---:|---:|
| DPO | 0.652 to 0.889 | 0.276 to 0.531 | 4.462 to 13.199 | 12/12 |
| GRPO + lr decay | 0.872 to 0.972 | 0.496 to 0.861 | 2.211 to 5.695 | 12/12 |
| Self-distillation | 0.908 to 0.979 | 0.426 to 0.668 | 0.344 to 0.591 | 0/12 |

L15.H11 ranks first among tested heads in 5/12 DPO, 12/12 GRPO and 11/12 self-distillation cells. L23.H1 ranks first in only 2/12 DPO cells. These are ranks within each adaptively selected test set, not ranks across the model.

For DPO combat/implicit, L23.H1 has true-intervention ranks 1/46, 2/47 and 2/44 for seeds 42, 43 and 44. A claim that it is always the new leading head is not supported.

Each range spans three seeds and four contrasts. These are descriptive ranges, not confidence intervals.

## Interpretation and limits

- one shared SFT-v3 checkpoint.
- Canonical SFT-v3 |effect| >= 0.5; seed 42; 64 pairs, gender 45; identical across all arms/seeds.
- Pearson across 64 real mixer/MLP suffix-patching effects.
- Descending signed real effect only among each run's AtP-selected plus control tested heads; not all-model rank.
- Existing candidate-bootstrap 95% CI is on share_removed, not logits per pair.
- p50/p99 of absolute logit effects across 450 aligned CV-job pairs per contrast; ratios to shared SFT; descriptive ranges, not CIs.
- HF interpretation loads the original Base and activates [SFT-v3, last-stage] adapters together; behavioral serving uses successively merged weights. Algebraic adapter addition is not evidence of numerical equivalence across these execution paths.
- Patch baselines and full-grid margins differ under their different batching contexts; see patch_vs_full_margin_diagnostics. Do not pool their per-pair margins as numerically interchangeable measurements.
- Replicated checkpoints on one development benchmark do not establish causal routes or external generalization. Neither p99 effects nor L23.H1 results identify the cause of high-hire-rate behavioral deterioration.
- AtP columns remain explicitly separate from real intervention columns. The tested head set changes by arm and seed.
- Patch versus full-grid baselines differ by up to 15.69 logits, including fresh bound caches. Selected tokens decode to the same hashed prompt strings. The two passes use different batch construction. This is a measured execution-context discrepancy, not proof that the difference is harmless noise. Do not substitute full-grid baselines into patch-effect denominators.
- The internal final-pre-norm swap control differs from B's recorded margin by at most 0 logits. This checks the intervention readout path; it does not resolve the separate cross-pass discrepancy.
- The old SFT-v3, DPO seed 42, GRPO seed 42 and self-distillation seed 43 patch caches lack historical adapter-identity metadata. Self-distillation seed 42 reused legacy combat/implicit and religion/implicit caches; its other two contrasts were bound before new computation. Thus 18/40 caches retain a historical identity limitation and 22/40 were bound before computation. All current full-grid margin fingerprints match current weights. Raw effects reconcile with exported CSVs, and every pair selection matches, but those checks cannot prove historical cache identity.

## Reproduce

```sh
/ritom/scratch/cscs/zhang/xw/venv/bin/python scripts/summarize_mechanism_seeds.py --input-root /ritom/scratch/cscs/zhang/xw
```

The script writes only this aggregate report directory. Source-file SHA256 values and current adapter hashes are recorded in provenance.json. No CV text or candidate identifiers are exported.

## Files

- component_profiles.csv: 36 genuine intervention profile correlations, cross-checked against canonical component_summary.csv.
- target_heads.csv: L15.H11 and L23.H1, real effects, share CIs, ranks within tested heads, and separate AtP estimates.
- head_verification.csv: tested counts, controls, top tested heads and subset-only AtP agreement.
- tail_effects.csv: 36 all-pair distributions, recomputed from current margin files and cross-checked against sensitivity.csv.
- selection_checks.csv and raw_summary_checks.csv: canonical selection and raw-to-CSV reconciliation.
- summary.json and provenance.json: aggregate findings, scope, hashes and historical identity limitations.

## Subsequent baseline replay

The CPU snapshot above could not isolate the cross-pass margin discrepancy. A later [GPU batch-context replay](batch_diagnostic/README.md) exactly reproduced both historical baseline passes for eight deliberately selected DPO seed-43 combat/explicit pairs. Repeated contexts gave identical margins, while switching contexts reproduced the 15.69-logit maximum difference. The follow-up documents the selection and runtime limits; it does not rerun or validate intervention rankings across batching contexts. The original CPU summary and its frozen source hashes remain unchanged.
