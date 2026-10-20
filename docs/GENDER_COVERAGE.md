# Gender coverage: more gender attributes per pair, pi1's mistakes on them

Status: pre-registered 2026-10-19 before any GPU job (commit d9b5555); run the same day.

**Conclusion.** Fact: on 11,916 new gender rows pi1 makes attribute-tipped mistakes at the same
rate as on the same pairs' original gender rows (1.04% vs 1.07%, difference -0.03 pp, 95% CI
-0.20 to +0.14), and 102 of the 124 new error rows sit on pairs that already erred.
Interpretation: pi1's remaining attribute flips belong to a few fragile CV/job pairs, not to
particular gender attributes, so more gender rows add little new preference data.
Unknown: whether any inserted line, not only a protected attribute, flips those pairs.

| Field | Value |
|---|---|
| Code | branch xinyang/gender-coverage, d9b5555 (logged as `commit` in each job log); the comparison script and this card in the follow-up commit |
| Config | configs/generation/negatives_v2_onpolicy_gender_pi1_gh200.yaml (`add_attributes` gender k=2 seed 20261019, `split_of` onpolicy-it2) |
| Student | pi1 = outputs/dpo/qwen3.5-9b_en_only_dpo_v2neg_r2em/checkpoint-100, merged with its SFT parent (outputs/merged/qwen3.5-9b_en_only_dpo_v2neg_r2em--checkpoint-100) |
| Data | $HBM_OUTPUT_ROOT/artifacts/semisynthetic-v2/onpolicy-gender-pi1 (rows, prompts, raw/samples, selected, candidates, report.json, direction_report.md, check_all/direction_report.md) |
| Jobs | sample 5098345 (11 min), judge 5098495 (6 min), check 5098780 (2 min), all on `normal` with a 30-min limit |
| Comparison | scripts/compare_onpolicy_rows.py, base onpolicy-it2 (gender rows) -> reports/gender_coverage/{summary.json,by_attribute.csv,pairs.csv} |

## Setup: the local dry run (written before running)

Question. Can the pool's gender coverage be evened out with new gender rows that reuse the
pair's existing v1 chosen text (no new teacher calls), and how many rows do the repo's own
filters lose doing it?

Why. v1 draws 2 of 20 gender attributes per pair (synth.py:206-207), so each gender attribute
has 526-624 pool rows against ~2,400 for each military attribute. synth.py:194-195 assumes
"thousands" per attribute.

Construction.
- Input: the it2 copy of the pool, 35,826 rows over 2,992 pairs (scratchpad wf/it2/rows.parquet).
- Per pair, draw K new gender attributes uniformly from the 18 it does not have yet, with
  numpy default_rng(20261019); this rng is separate from v1's seed-42 stream. Primary K = 2,
  which puts gender at 4 of 20 per pair, about religion's coverage. Variant K = 6 gives 8 of
  20, about military's.
- Each new attribute gets both conditions, as in v1.
- chosen decision = the pair's reference, as in v1. chosen text = one of the pair's existing
  gender rows in the same condition, picked by the same rng. Texts with a gendered pronoun
  (she/her/he/him...) are never reused. A missing source is counted, not filled.
- Profile via synth.build_profile and audit prompt via onpolicy.attributed_prompts, so the
  new rows render exactly like pool rows.

Checks run locally.
- The repo's filters.filter_invariant on the reused text under the NEW attribute: decision
  drift, length, language, attribute leak, meta commentary. Drop reasons per attribute.
- assert_no_leakage against data/benchmark/holdout_ids.json.
- Validation split: new rows on the 359 validation candidates (negatives_v2.validation_candidates
  with v1's held-out ids, 0.05, seed 42) are counted and kept apart.
- D1 chosen side (decision = reference) and D6 (chosen hire share) from direction_check.
- Coverage per gender attribute before and after, and text reuse counts.
- Blind coherence read: reused rows mixed with original pool rows as controls; readers do not
  know which is which and flag rationales that refer to gender or do not fit the profile.

Not computable without a GPU. D2 covertness, D3 separability, D4 student preference and D5
overlap need the student's rejected samples on the new prompts (one Daint sampling job).

Pass criteria, fixed now.
- Filter yield >= 95% overall and >= 85% for every gender attribute. An attribute below 85%
  needs whole-word leak matching before its rows are used.
- Coherence: reused rows flagged at most 2 pp more often than the original controls.
- D6: new rows' hire share within 1 pp of the pool's 14.16%.

The dry run's outputs stayed in the session scratchpad; nothing was written to Daint.

## Addendum, 2026-10-19, before any GPU job: pi1 sampling on the new rows (user approved)

Run. configs/generation/negatives_v2_onpolicy_gender_pi1_gh200.yaml on Daint, branch
xinyang/gender-coverage: the 11,916 new gender rows (same builder, now in
onpolicy.add_attribute_rows, reproducing the dry run exactly) plus a fresh attribute-free
prompt per pair, pi1 (r2em checkpoint-100, merged with SFT) at k=8, T=1; tipping, covert
filters and the 122B judge exactly as it2; split from it2's rows (split_of). One debug job per
stage, one at a time: sample, judge (+ GPU-free D1-D3, D5-D7), check with
CHECK_PAIRS=candidates (D4 with pi1 and SFT on all judged pairs).

Expectation, from it2's gender slice (same pairs, original attributes: 92 judged pairs, 1 hire,
on ~11,900 gender rows): roughly 60-140 judged pairs, almost all reject-reference, most on
pairs that already erred in it2.

Readouts.
- Primary: pi1's tipped-with-wrong rate on the new attributes against its rate on the same
  pairs' original gender attributes (it2 samples), and how many new pairs erred.
- D1 validity, D2 covertness (lexical and judge allusion), D5 overlap, D4 pi1 and SFT
  preference, against it2's numbers where it2 has them.
- D3 is declared NOT assessable here. The candidates are one decision class (or nearly), so
  negatives_v2.separability falls back to the overall AUC, which is the within-decision AUC
  and high by design; with 0 hire pairs the report prints it as a failing "balanced AUC".
  Only auc_within_decision is quoted, beside it2's reject-only value (0.999), descriptively.
- Not available for comparison: it2 never ran the check stage, so there is no it2 D4.

Stop. A stage that fails twice is not resubmitted; report instead.

## Dry-run result (local, 2026-10-19)

- 11,916 of 11,968 planned rows kept (99.6%). 8 rows lost: 2 pairs with no gender rows. 44 dropped by the leak filter's 5-character stems, all false positives (android, intermediate/internship/international, translate/transformation): Intersex 96.7%, Androgynous 97.0%, Transgender 99.0%, every other attribute 100%.
- Gender rows per attribute 526-624 before, 1,139-1,254 after. Hire share of the new rows 0.1415 (pool 0.1416). 10,205 rows on training candidates, 1,711 on validation candidates.
- No new chosen texts: 13,480 distinct before and after; in 77% of (pair, condition) cells both new attributes carry the same text.
- Independent recount reproduced every number; prompts render byte-identically to the sampler's; a blind read flagged 0 of 120 reused rationales and 2 of 60 original ones (both a generic singular 'they').


## Result

Sampling (sample_report.json): 11,916 rows x 8 samples; pi1 right 86.0% with the attribute and
86.2% without; 1,688 rows with any wrong sample, 124 attribute-tipped with a wrong sample, all
124 kept by the covert filters (raw attribute or group mention 0.3%).

Judge (report.json): 105 of 124 pass the rejected-side judge (18 ungrounded, 1 contradicts),
96 after the chosen re-check (9 ungrounded); split 74 train / 22 validation.

Against the same pairs' original gender rows in it2 (reports/gender_coverage/summary.json, pair
bootstrap, 10,000 draws):

| Slice | New attributes | Original attributes (it2) | Difference, pp [95% CI] |
|---|---|---|---|
| All | 124 / 11,916 = 1.04% | 128 / 11,918 = 1.07% | -0.03 [-0.20, +0.14] |
| Explicit | 73 / 5,957 | 77 / 5,960 | -0.07 [-0.30, +0.17] |
| Implicit | 51 / 5,959 | 51 / 5,958 | -0.00 [-0.20, +0.20] |
| Reference reject | 118 / 10,230 | 126 / 10,229 | -0.08 [-0.27, +0.12] |
| Reference hire | 6 / 1,686 (2 pairs) | 2 / 1,689 (1 pair) | +0.24 [0.00, +0.65] |

Pairs with an error row: 54 on new attributes, 57 on original ones, 42 in both. 102 of the 124
new error rows are on pairs that already erred on their original gender rows
(reports/gender_coverage/pairs.csv; per attribute in by_attribute.csv).

Data checks on the 96 judged pairs (direction_report.md; D4 in check_all/direction_report.md):

| Check | Value | Reading |
|---|---|---|
| D1 validity | 0 chosen off reference, 0 rejected not flipped | pass |
| D2 covertness | lexical 0.000, judge allusion 0.000 | pass |
| D3 separability | not assessable (pre-registered): 93 of 96 pairs are reject-reference, so the printed "balanced AUC 0.700" rests on 6 pairs; within-decision 0.992 (it2 0.999) | n/a |
| D5 overlap | mean Jaccard 0.136 | no flag |
| D6 balance | chosen hire share 0.031 (3 of 96) | fails [0.4, 0.6]; as expected for this pool |
| D4, pi1 | prefers chosen 15.6% (per token 11.5%), decides the reference 7.3% | pi1's own confident mistakes; 84% informative |
| D4, SFT | prefers chosen 50.0%, decides the reference 47.9% | SFT is split on the same pairs |
| D7 yield | 96 / 11,916 = 0.8% | it2 305 / 35,826 = 0.85% |

D4's band [0.5, 0.9] was set for teacher-written negatives scored by SFT; on-policy negatives of
the scoring model itself sit below it by construction.

## Reading

- What the evidence supports: the extra gender attributes do not expose new weaknesses of pi1.
  The new error rows repeat it2's error pairs, and pi1 errs on those pairs whatever gender line
  is added. pi1 is far more confident in these mistakes than SFT (D4 7.3% vs 47.9% reference
  decisions), which matches docs/ITERATIVE_DPO.md: DPO sharpened the remaining flips.
- What it does not support: that gender coverage was limiting training, or that these 96 pairs
  would help. 93 of 96 teach "reject" on reject-reference CVs, the direction that made rounds 1
  and 2 stricter, and most of their chosen texts already appear in it2's candidates.
- Cheapest next check, if wanted: insert a neutral line of the same length (no protected
  attribute) into the error pairs and sample pi1 again; one short GPU job tells whether the
  flips are about attributes at all.

## Caveats

- One sampling seed for every request (eval/backends.py `generate_samples`: one SamplingParams
  with seed 42 and n=k for all prompts). Prompts that differ only in the attribute line share
  their random stream, so samples are correlated across a pair's rows and between runs; an
  exploratory read found 16-17% of same-index samples identical across attributes of a pair,
  and most attribute-free samples identical to it2's. Most flips here are 8/8 against 8/8, so the
  readout stands, but the k samples are not independent draws across rows. This holds for
  round 1 and it2 too.
- Single seed, single student; the hire stratum has 3 pairs.
- The teacher that judged is the Daint teacher, which differs from v1's (docs/SFT_V3_ANCHORS.md
  on branch xinyang/sft-v3-anchors); the same judge was used for it2, so the comparison holds.

## Reproducing

    REPO=$XW/hiring_bias_mitigation_at-gender \
    CONFIG=configs/generation/negatives_v2_onpolicy_gender_pi1_gh200.yaml \
      scripts/cscs/submit.sh -p normal -t 00:30:00 scripts/cscs/onpolicy_full.sbatch sample
    # then `judge`, then CHECK_PAIRS=candidates ... `check`
    python scripts/compare_onpolicy_rows.py --base artifacts/semisynthetic-v2/onpolicy-it2 \
      --new artifacts/semisynthetic-v2/onpolicy-gender-pi1 --group gender \
      --out reports/gender_coverage
