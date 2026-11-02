# Decision geometry and affected units

CPU analysis of existing scores; no new model inference or training. The analysis covers SFT-v3 and three seeds each of DPO, GRPO R4, and self-distillation. Each condition uses all 31,050 scored prompts, including the 450 attribute-free prompts when calibrating the global hire rate. Instability is evaluated on 2,700 counterfactual sets.

The 51-point curve uses the exact grid and integer-count rounding in `scripts/final_table.py`, from 10% through 35% in 0.5 percentage-point steps. The additional common point hires exactly 6,990 prompts, or 22.51207729468599%.

## Findings

The tail change survives rank normalization. Across the twelve seed-by-contrast observations, HF absolute-effect p99 ratios to SFT-v3 are 4.462–13.199 for DPO and 2.211–5.695 for GRPO. Their empirical-percentile effect p99 ratios are still 2.489–4.365 and 1.516–2.366. vLLM gives the same direction. A single strictly increasing transformation of SFT's margins, preserving ties, therefore cannot explain these effect changes. This statement concerns the evaluated prompt grid, not population calibration.

Large effects do not identify the same decision risks at every working point. Each contrast's top 1% contains five pairs. At the common point, 4–13 of DPO's 20 HF contrast-pair observations cross the cut per seed; the vLLM counts are 7–11. At all nine 31–35% grid points, none of those DPO tail observations crosses the cut in either numerical path: zero weighted crossings among 540 contrast-pair-rate observations per path. These 540 observations repeat the same pairs across rates and are not independent samples.

Nevertheless, DPO's complete 2,700-set curve deteriorates throughout that high-rate band under vLLM: all 27 seed-by-rate changes are positive, from +0.371 to +4.521 pp. HF has 26 positive changes and one −0.037 pp change at seed 42, 31%. At 35%, the three-seed mean change is +3.458 pp for vLLM and +2.889 pp for HF. The four binary contrasts therefore do not explain the entire curve. The complete six-cell family/cue decomposition is supplied separately and closes exactly to the global result. All six cells have a positive mean contribution to DPO's 35% degradation; gender explicit and implicit together account for about half.

Affected CVs recur more consistently than individual unstable sets. At the common point, vLLM fuzzy Jaccard ranges over the three seed pairs are:

| Recipe | Set | CV–job pair | CV |
|---|---:|---:|---:|
| DPO | 0.172–0.275 | 0.423–0.640 | 0.440–0.667 |
| GRPO R4 | 0.299–0.347 | 0.415–0.513 | 0.488–0.596 |
| Self-distillation | 0.533–0.544 | 0.749–0.797 | 0.736–0.877 |

Thus, low overlap of `(pair, family, cue)` sets does not imply equally low overlap of affected people. These ranges are three descriptive overlaps, not confidence intervals.

The original 35 borderline pairs represent 33 CVs and 33 jobs. At the common point they contain 77.8–83.6% of DPO's remaining unstable-set mass, 84.8–90.1% of GRPO's, and 93.3–95.0% of self-distillation's. They were defined by SFT-v3's attribute-free absolute margin below 2 at its zero cut. At 35%, they contain only 2.5–10.8% of DPO's unstable mass. The old zero-cut band is not a universal description of difficult cases at other working points.

Deleting one CV or job with cuts fixed does not reverse any DPO seed's common-point sign, and every vLLM DPO high-rate change remains positive after any such deletion. Some much smaller changes are sensitive: the positive GRPO seed-44 common-point change reverses after deleting two individual CVs or one job; the negative self-distillation seed-42 change reverses after deleting one CV or one job. This is an influence check, not a new bootstrap, confidence interval, or unseen-data test. The existing poster intervals already cluster by CV.

## Numerical and statistical scope

The `backend` column abbreviates two complete numerical paths. Legacy HF `interp.spec` activates the SFT and final PEFT adapters together on the original base and reads `margin_fp32`. vLLM serves sequentially bf16-merged checkpoints and uses canonical op-margin-v2 margins. Differences include model representation and rounding as well as backend. They are not a controlled backend-only experiment. New experiments fixing a merged-SFT base are a different condition from these legacy HF scores.

Each model and numerical path gets its own threshold. vLLM uses the canonical 1/1024 snapping grid; HF remains unsnapped. No HF effect is compared with a vLLM numeric threshold. HF metadata also record different batch sizes across some original and follow-up runs.

Absolute effect is the difference in decision margins between the two cue values, in magnitude. Rank effect is the absolute difference in full-grid empirical midrank divided by 31,050. The latter is invariant to strictly increasing transformations that preserve ties. Quantiles use linear interpolation over 450 pairs per contrast. Every top-1% set includes all pairs tied at or above its p99 boundary; p95, absolute p99, top-set size and share of total effect are also reported.

Threshold ties use a shared mixture of the strict and inclusive cuts. Pair/CV unions are computed separately in each branch before mixing. Taking the maximum of already-fractional set flags would undercount units whose sets alternate instability across the two cuts. Across seeds, Jaccard uses `sum(min(weights))/sum(max(weights))`; this is descriptive fuzzy overlap, not a joint probability. Added and removed masses are positive and negative membership differences, not independently observed causal transitions.

All results remain exploratory measurements on the development benchmark. They do not establish whether an individual hire/reject decision is correct or whether risk generalizes to new CVs, jobs, or cue templates.

## Files and reproduction

- `effect_summary.csv`: p50/p95/p99 of absolute and rank effects, ratios to SFT, top-pair counts/contributions, effect-rank correlation and direction changes.
- `cuts.csv`: every separately calibrated cut and tie weight.
- `threshold_geometry.csv`: four-contrast crossing counts, tail crossings and distances at all rates.
- `common_rate_pair_features.csv`: numeric margins, effects, distances and crossings for the common point, using hashed IDs only.
- `global_curve.csv`, `family_cue_curve.csv`, `high_rate_pair_contributions.csv`: full-grid curve and closed contributions by family/cue or pair.
- `common_rate_overlap.csv`, `common_rate_burden.csv`: set/pair/CV overlap and affected burden with shared tie handling.
- `leave_one_out.csv`, `leave_one_out_summary.csv`: fixed-cut CV/job deletion influence at the common point and all nine 31–35% points.
- `manifest.json`: definitions, source and metadata hashes, code hashes, representation caveat and all generated CSV hashes. Source metadata contain research configuration, not CV/JD text.

Run from the repository with the existing CPU environment:

```sh
rtk proxy env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /ritom/scratch/cscs/zhang/xw/venv/bin/python scripts/analyze_tail_decisions.py \
  --input-root /ritom/scratch/cscs/zhang/xw \
  --out reports/research_followup/decision_geometry
```

Read `rate_label` as a string when loading CSVs, because it combines numeric grid labels and `sft_common`. All eight boundary-case tests pass locally and remotely. The nine canonical common-point deltas and borderline shares match the existing poster follow-up table. Family/cue contributions close for all 1,040 model/path/rate conditions; high-rate pair contributions close for all 162 post-model/path/rate conditions. Output logical keys are unique. No raw CV, job description, or prompt text is exported. Hashed IDs permit linkage across these tables and are not a claim of irreversible anonymization.
