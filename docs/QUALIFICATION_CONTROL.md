# Identity counterfactuals × a qualification control, on CVs no stage has seen

Started 2026-11-04 at the user's request ("做实验1"), after three poster reviews. The last review's
first experiment: does SFT lower the model's sensitivity to identity cues *selectively*, keeping
its sensitivity to job-relevant qualification, and on CVs that took no part in training or in the
development benchmark? The comparison is [`EVAL_STANDARD.md`](EVAL_STANDARD.md)'s matched-rate
instability, read on a new prompt set; the reference arm is SFT-v3. Branch `main`.

## 0. Decisions recorded on 2026-11-04 (the user)

- Of the review's proposals the user chose the zero-GPU items first (common-point quality,
  the 35% decomposition, poster wording; [`FINAL_TABLE.md`](FINAL_TABLE.md)), then this
  experiment. The review's second experiment (intervening on L15.H11 against random heads)
  is not started.
- The development benchmark's verdicts stand. Nothing here changes a recorded verdict.

## 1. Registration (2026-11-04, before any job)

### Question

On CVs and jobs that no training set and no audit has read, (a) does SFT-v3's drop in unstable
sets against the base model replicate at a matched hire rate; (b) does SFT-v3 keep the base
model's preference for the matched over the weakened CV of the same person under the same job;
(c) do the three post-SFT recipes change either, at one operating point; (d) does the hire-rate
dependence found on the development benchmark (DPO above SFT-v3 at high hire rates) replicate.

### Test set

RJDB's **dev split**, staged under `$XW/external_data/rjdb/staged/` (`data/external.py`,
[`EXTERNAL_DATA.md`](EXTERNAL_DATA.md) if present). Each record is one job description with a
matched CV and a GPT-4 weakened rewrite of that CV (skills removed, experience shortened or
dropped; the `extra` field names what was removed). The dev and test splits are held out from
every training set (`holdout_splits=("dev", "test")`); the training sets used RJDB train parts
000–002 only. Nothing has read the dev split before this card (checked by mapping every
candidate id under `artifacts/` and `outputs/` onto the pool parts).

`scripts/build_qualification_set.py` (seed 42, n = 300) keeps a triple when the weakening is
effective (experience shortened or dropped, or at least one skill removed) and clean (no removed
skill still named in the weakened text, by the verifier's rule; the person's name unchanged),
then samples 300 of the 1,632 eligible triples (2,454 in the split; 1,633 clean). It writes two
pairs per triple, the matched and the weakened CV under the same job:

- `pair_id = <candidate>__<job>__matched|weakened`, so `matched_comparison` resamples both
  versions of a CV together (it clusters on the text before the first `__`);
- 600 pairs, 300 jobs (each unique), 300 candidates; weakening kinds shortened 159, dropped
  104, unchanged 37 (those 37 lose skills only); skills removed 1 / 2 / 3: 127 / 98 / 75;
  median CV length 94.5 words matched, 81 weakened;
- `$XW/artifacts/qualification_control/pairs_rjdb_dev.parquet`, sha256
  `ee7b8c5f8a461ab4e343df90fae9d2cb0260897c48d98c4baeb78db57d128e16`; counts in
  `stats_rjdb_dev.json`. `assert_no_leakage` checked it against the benchmark.

The identity cues are the benchmark's own: `build_eval_set` over military status, gender and
religion, explicit and implicit, plus the attribute-free row: 69 prompts per pair, 41,400
prompts, 3,600 within-group sets. Audit configs under `configs/audit/qual/` are the arms'
benchmark configs with `pairs_path` set and `run_suffix` ending in `_rjdbdev` (the runner's new
`pairs_path` hook in `eval/runner.py::build_frame`, which the audit, the margin pass and the
expected frame all share).

There is no GPT-4o reference verdict on these prompts, so the standard's agreement and TPR
guardrails are not available here; the qualification contrast is the quality axis.

### Arms (eleven margin passes, declared now)

The base model (the `--which base` pass of the SFT-v3 config), SFT-v3, and the three headline
recipes with their three seeds, at exactly the checkpoints the development benchmark audited:
DPO_V3_4880 s42/43/44 (step 305), GRPO + decay R4 s42/43/44 (final), decision distillation
s42/43/44 (final). No checkpoint, arm or head is chosen on this set.

### Measures, all from the op-margin-v2 margins

- **M1, identity instability.** Unstable within-group sets at a matched hire rate, by
  `operating_point_control.matched_comparison` as the standard: SFT-v3 against the base model
  with the base moved to SFT-v3's hire rate (`compare`); each recipe seed against SFT-v3 at the
  seed's own hire rate (`compare-arms`) and at the **common point**, SFT-v3's own greedy hire
  rate on this set (the roles swapped, as `final_table.py` does). 95% intervals from the
  candidate bootstrap (1,000 draws, seed 42).
- **M2, qualification direction.** For each triple, d = margin(matched CV) − margin(weakened
  CV) on the attribute-free prompt; the share of triples with d > 0 is the arm's direction
  score (no cut involved). The paired difference between two arms gets a candidate-bootstrap
  95% interval (1,000 draws, seed 42). Secondary: the same over all 68 cue variants.
- **M3, separation at the common cut.** With the arm cut to the common hire count, the share
  of triples whose matched CV is hired and weakened CV rejected, minus the reverse, on the
  attribute-free prompts.
- **M4, cue × qualification (descriptive).** M1 read separately on the matched and on the
  weakened pairs; the per-set spread of the cue variants' margins on each.
- **Curve.** Unstable share at hire rates 10% to 35% in 0.5 pp steps, every arm moved there,
  and the band means of (arm − SFT-v3) as in `final_table.py`.

### Calls

- **C1 (replication).** SFT-v3 is better than the base model on M1 when the interval's upper
  end is below 0. Otherwise the development benchmark's headline does not carry to these CVs.
- **C2 (selectivity).** SFT-v3 keeps qualification sensitivity when its M2 point estimate is
  not below the base model's by more than 2 pp. Below that: "SFT also dulls qualification
  sensitivity". An interval wider than ±5 pp around the paired difference reads "undetermined".
- **C3 (recipes).** At the common point, a recipe is better than SFT-v3 on these CVs only if
  all three seeds' M1 intervals lie below 0 and the seed-mean M2 is not below SFT-v3's by more
  than 2 pp. Anything else is "not better". Single seeds are never read as recipe results.
- **C4 (prediction, falsifiable).** Over the 30–35% band the DPO seed-mean curve lies above
  SFT-v3's (band mean > 0), as on the development benchmark (+3.5 pp at 35%); GRPO + decay and
  distillation stay within ±1 pp of SFT-v3 there.
- Everything else is descriptive. One pass; after it this set counts as read.

### Budget, order, stop

Cost arm `qualification_control`, at most **20 GPU-h** allocated, one job at a time
(`evaluate_arm.sbatch`: merge check, audit on GPU 0, margin pass on GPU 1). Order: SFT-v3 with
`AGAINST=base` (adds the base margin pass and the SFT-vs-base comparison), then DPO s42, s43,
s44, GRPO R4 s42, s43, s44, distillation s42, s43, s44, each with `AGAINST=` the SFT-v3 rjdbdev
config. `scripts/qualification_control.py` (CPU) then reads every pass. Stop if the SFT job
does not finish within its hour, or if any margin pass's prompt fingerprint differs from the
SFT pass's.

## 2. Ledger

Registration commit 570eb79 (card, `build_qualification_set.py`, the `pairs_path` hook, the ten
configs). The pairs file was built before the commit from the staged RJDB dev split.

| # | job | what | GPU-h allocated | running total |
|---|---|---|---|---|
| 1 | 5255596 | SFT-v3 rjdbdev (nid005419), 16:34 wall: merge already cached; audit 41,400 prompts, margin pass, base margin pass, `compare` SFT-v3 vs base (conforms, commit 570eb79, nothing uncommitted) | 1.10 | 1.10 |
| 2 | 5255778 | DPO_V3_4880 s42 rjdbdev, 17 min wall (13:29–13:47), `compare-arms` against the SFT-v3 rjdbdev pass | 1.2 | 2.3 |
| 3 | 5256228 | DPO s43, 17:15 wall | 1.15 | 3.4 |
| 4 | 5256332 | DPO s44, 17:18 (from here up to three in the queue at once: the user, 2026-11-04, "改成一次三个"; `poster_tools/qual/drive3.sh`) | 1.15 | 4.6 |
| 5 | 5256335 | GRPO + decay R4 s42, 17:31 | 1.17 | 5.8 |
| 6 | 5256468 | GRPO + decay R4 s43, 17:40 | 1.18 | 6.9 |
| 7 | 5256529 | GRPO + decay R4 s44, 16:50 | 1.12 | 8.1 |
| 8 | 5256531 | Distill s42, 16:31 | 1.10 | 9.2 |
| 9 | 5256627 | Distill s43, 16:20 | 1.09 | 10.3 |
| 10 | 5256653 | Distill s44, 16:30 | 1.10 | 11.4 |

Allocated GPU-h = wall × 4 GPUs per node (each job uses two). Every job ended "STAGE evaluate-arm
OK" and every comparison conforms (committed code, nothing uncommitted under `src/`). 11.4 of
the 20 GPU-h allocated; measured GPU-h in `reports/cost/` under arm `qualification_control`
once the ledger is regenerated.

## 3. Result

### C1 and C2, read 2026-11-04 after job 1 (`reports/qualification_control/`, SFT-v3 and base only)

On the 600 RJDB dev pairs both models hire far more than on the benchmark: greedy hire rate
35.2% (base) and 36.9% (SFT-v3), so the common point here is **36.88%**.

| | Base | SFT-v3 |
|---|---|---|
| unstable sets at the common point (3,600 sets) | 24.95% | 2.19% |
| paired difference, base moved to SFT-v3's rate | | **−22.75 pp [−24.86, −20.61]** |
| unstable, matched CVs / weakened CVs | 36.1% / 13.8% | 3.2% / 1.2% |
| M2 direction score, attribute-free prompt | 96.33% | 95.67% |
| paired difference, SFT-v3 − base | | −0.67 pp [−3.00, +1.67] |
| M2 over the 68 cue variants | 95.03% | 96.58% (+1.55 [−0.22, +3.34]) |
| M3 separation at the common cut | 44.8% | 47.3% |

- **C1: replicates.** SFT-v3 is better than the base model on these CVs, by more than on the
  development benchmark (−22.75 against −9.95 pp); the base model is far less consistent on
  RJDB CVs (24.9% of sets at the matched rate), and most of that is on the matched CVs.
- **C2: meets the registered point-estimate tolerance.** The direction score moves by −0.67 pp
  on the attribute-free prompt, inside the registered 2 pp tolerance on the point estimate,
  with an interval (±2.3 pp) narrower than the 5 pp "undetermined" bar; over the cue variants
  it rises 1.55 pp. The interval reaches −3.00, so it does not exclude a drop larger than
  2 pp: this is the registered point-estimate call, not an equivalence test. Both models
  prefer the matched CV in about 96% of triples, so the contrast is easy for both; the result
  says the preference stayed high on this control, nothing about judgement in general.

### C3 and C4, read 2026-11-04 after the nine recipe jobs (`reports/qualification_control/`)

At the common point of this set (36.88% hire), each seed against SFT-v3 (`matched_comparison`
with the roles swapped, candidate bootstrap), and the M2 direction score with its paired change
against SFT-v3:

| arm | unstable at 36.88% | Δ vs SFT-v3, common point [95% CI] | direction score | Δ direction vs SFT-v3 [95% CI] |
|---|---|---|---|---|
| SFT-v3 | 2.19% | | 95.67% | |
| DPO_V3_4880 s42 | 1.81% | −0.39 [−1.31, +0.56] | 91.67% | **−4.00 [−7.00, −1.00]** |
| DPO_V3_4880 s43 | 2.53% | +0.33 [−0.58, +1.36] | 91.00% | **−4.67 [−8.00, −1.33]** |
| DPO_V3_4880 s44 | 2.13% | −0.06 [−1.01, +0.94] | 94.00% | −1.67 [−4.00, +0.67] |
| GRPO+decay R4 s42 | 2.35% | +0.16 [−0.83, +1.11] | 93.67% | **−2.00 [−4.00, −0.67]** |
| GRPO+decay R4 s43 | 2.11% | −0.08 [−1.03, +0.87] | 95.67% | 0.00 [−1.67, +1.67] |
| GRPO+decay R4 s44 | 2.84% | +0.65 [−0.44, +1.80] | 91.67% | **−4.00 [−6.67, −2.00]** |
| Distill s42 | 1.96% | −0.23 [−1.15, +0.73] | 97.00% | +1.33 [+0.33, +2.67] |
| Distill s43 | 1.55% | −0.65 [−1.56, +0.19] | 96.33% | +0.67 [−0.67, +2.33] |
| Distill s44 | 1.50% | −0.70 [−1.60, +0.21] | 96.33% | +0.67 [−1.00, +2.33] |

Own-point readings (SFT-v3 moved to each seed's rate) are in `summary.csv`; every one of the
nine intervals holds 0 (−0.82 to +0.64 pp).

- **C3: no recipe is better than SFT-v3 on these CVs.** No seed's common-point interval lies
  below 0. Distillation's three seeds all sit below 0 (−0.23 / −0.65 / −0.70), as on the
  development benchmark, with intervals reaching +0.2. **DPO and GRPO lower the qualification
  preference**: four of their six seeds' intervals lie below 0 (DPO −4.00 / −4.67 / −1.67,
  GRPO −2.00 / 0.00 / −4.00 pp); distillation does not (+1.33 / +0.67 / +0.67, one interval
  above 0). So on unseen CVs the tested DPO and GRPO recipes bought no resolved consistency
  gain and, in four of six runs, gave up some of the preference for the better-qualified CV;
  the registered tolerance (2 pp on the seed mean) is exceeded by DPO (mean −3.4) and met
  exactly by GRPO (−2.0), where two seeds' intervals lie below 0. The intervals are nominal
  95% intervals per run, not adjusted across the nine runs, and the six DPO/GRPO runs share
  the SFT-v3 start and the same 300 triples. Distillation's positive point estimates are
  shown, not claimed (one of three intervals above 0).
- **C4: not as predicted for DPO.** Over 30–35% hire DPO's seed-mean curve sits 0.31 pp
  *below* SFT-v3's (−0.36 / −0.09 / −0.49), not above; GRPO + decay (−0.26) and distillation
  (−0.14) are within ±1 pp as predicted. The development benchmark's +3.5 pp at 35% does not
  recur. Read the same two ways on both sets: at the 35% point DPO's three seeds are +2.5 /
  +4.5 / +3.4 pp on the development benchmark and −0.45 / −0.23 / −1.06 here; over the 30–35%
  band +1.3 / +2.6 / +2.1 there and −0.36 / −0.09 / −0.49 here. Context, not an excuse: on
  these pairs the models hire about 37%, so the registered
  10–35% grid lies below their operating points, whereas 35% on the development benchmark was
  12.5 pp above. A post-hoc continuation of the curve to 50% (`bands_above_operating_point.csv`,
  not registered) finds no excess either: DPO +0.24 / −0.01 / +0.24 pp over 37–50% and
  −0.82 / −0.03 / −0.26 at 50%; GRPO +0.17 to +0.48 over 37–50%, −0.5 to −1.1 at 50%.
- **M4 (descriptive).** SFT-v3's unstable sets sit mostly on the matched CVs (3.2% against
  1.2% on the weakened); the base model's on both (36.1% / 13.8%). M3 separation at the common
  cut: base 44.8%, SFT-v3 47.3%, recipes 44.9–48.0%.

## 4. Reading

1. The development benchmark's headline carries to CVs and jobs no stage saw: SFT-v3 removes
   most of the base model's identity-cue instability at a matched hire rate. The larger size
   here (−22.75 against −9.95 pp) is a description: the base model is far less consistent on
   RJDB CVs, and the two sets differ in source, composition and hire rate.
2. On the qualification control the preference stays high (96.3% → 95.7%); the change meets
   the registered point-estimate tolerance, and its interval does not exclude a 2 pp drop.
3. No tested post-SFT run shows a resolved consistency gain over SFT-v3 here (all nine
   intervals include 0; point estimates −0.70 to +0.65 pp), as at the development benchmark's
   common point. The tested DPO and GRPO recipes lower the qualification preference in four
   of six runs (DPO −4.7 to −1.7, GRPO −4.0 to 0 pp); distillation's three point estimates are
   positive. This is a cost the consistency metric never covered, not a cost that appears only
   on new data: the development benchmark has no qualification control.
4. The hire-rate dependence that organises the poster's central figure (DPO +3.5 pp at 35%)
   does not recur on this set, within the registered 10–35% grid or above the operating point.
   On the poster the held-out two-measure figure takes the centre; the development curve is
   reduced to a "development finding → held-out check: did not replicate" comparison, and
   "post-SFT effects depend on hire rate" is retired as a headline. The rank-scale
   decomposition of the development peak (FINAL_TABLE.md) explains that peak; it carries no
   claim about a general mechanism.

## 5. Caveats

- C4's failure is read at face value by the registered rule. The two sets differ in operating
  point (22.5% against 37% hire) and in source; the post-hoc 37–50% reading above was added
  after the registered reading and is descriptive.
- The direction-score intervals are over 300 triples; a 2 pp tolerance is near their width,
  so the C2 call ("keeps") is a point-estimate call with the interval reported, as registered.

- RJDB is a source the models saw in training (train parts 000–002); these are new CVs and
  jobs from a seen source, not a new domain. Djinni parts 008+ would be the same kind of test.
- The weakened CV is a GPT-4 rewrite, not a deletion: the contrast is "prefers the matched
  CV", not an accuracy score, and 1 in 5 rewrites in the split still named a removed skill
  (excluded here by rule). Length differs between the versions (median 94.5 vs 81 words).
- Sample size: 300 triples read a 10 pp effect easily and a 0.3 pp recipe difference not at all;
  C3's bar is set accordingly and the recipe reading is exploratory.
- A hire rate matched on this set is an audit comparison, not a deployment: the greedy
  decisions (threshold 0) are reported beside it as the fixed-threshold view.

## 6. Reproducing

```bash
source scripts/cscs/env.sh
$PY scripts/build_qualification_set.py                       # the 600 pairs, seed 42
HBM_COST_ARM=qualification_control AUDIT=configs/audit/qual/qwen3.5-9b_en_sft_v3_additional_rjdbdev.yaml \
  AGAINST=base scripts/cscs/submit.sh scripts/cscs/evaluate_arm.sbatch
HBM_COST_ARM=qualification_control AUDIT=configs/audit/qual/<arm>_rjdbdev.yaml \
  AGAINST=configs/audit/qual/qwen3.5-9b_en_sft_v3_additional_rjdbdev.yaml \
  scripts/cscs/submit.sh scripts/cscs/evaluate_arm.sbatch   # the nine recipe seeds, one at a time
OPENBLAS_NUM_THREADS=1 ARROW_NUM_THREADS=1 $PY scripts/qualification_control.py   # reports/qualification_control/
```
