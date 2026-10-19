# Gender coverage: more gender attributes per pair, pi1's mistakes on them

Status: pre-registered 2026-10-19 before any GPU job. Results to follow.

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
