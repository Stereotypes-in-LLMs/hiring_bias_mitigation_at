# Pre-registration: does the v2 judge favour its own text?

Written 2026-10-19, before any control sample was judged. The job log prints this file's
sha256 and `meta.json` records it, so a later edit shows up. Card: `docs/JUDGE_AUTHOR_CONTROL.md`.

## Question and hypotheses

On hire-reference rows, the 122B teacher judge passes v1's teacher-written `chosen` in 94.9 to
98.7% of rows per cell and the SFT student's wrong `rejected` in 52.8 to 70.8%
(`reports/teacher_leakage/judge_by_cell.csv`, commit cc38e2e on `origin/xinyang/teacher-anchor-check`).
About nine in ten failures on the rejected side are `ungrounded`. The two sides differ in the
author (teacher or student) and in the verdict (right or wrong).

- **H-verdict.** The judge fails the rejections because a rejection of a qualified candidate
  has to claim gaps the profile does not show. Who wrote the text does not matter.
- **H-author.** The judge passes text written by itself, or in its own register, and fails
  student text, whatever the verdict.

## Control

The same judge (`negatives_v2.judge_prompt`, `filter_by_judge`, `min_plausibility` 3, the round's
own `judge_generation`, temperature 0, `max_new_tokens` 160, the same teacher config with TP4)
on the SFT student's **correct** attributed sample from the same rows. The author is held at
"student" and the verdict at "right".

- Rows: each round's judged rows, which are `selected.parquet` restricted to `judge_strategies`
  (`student_tipped`). That gives 5,589 rows for `onpolicy-full` and 2,111 for `onpolicy-hire32`,
  the same rows that `judge_by_cell.csv` counts.
- Control sample: per row, the first attributed sample in sample order (from
  `<round>/raw/samples`) whose decision is the reference and that passes the hard filters the
  negatives passed (`negatives_v2.filter_negatives`: parsable, English, length within 10 words of
  chosen, no attribute or group word, no meta-commentary). The rejected side was picked the
  same way among wrong samples, so the two student sides are symmetric. The filter matters here.
  Without it, a correct sample that names the attribute would fail the judge for `alludes`,
  which neither original side could.
- Comparison: on **matched rows** (rows that have a control sample), with all three sides on
  the same rows. Chosen and rejected use their October judgements
  (`<round>/raw/{student,chosen}_judge.parquet`).

## Measures

Per round × (group, condition, reference_decision) and pooled per round × reference_decision:
the pass rate of each side with a Wilson 95% CI, the count of each failure reason (unparsable,
alludes, ungrounded, contradicts_decision, implausible), coverage (the share of rows with a
control), and an exact McNemar test of control against chosen and against rejected.

**Position** = (control − rejected) / (chosen − rejected) on matched rows, with a 2,000-rep
paired percentile bootstrap over rows (seed 42). A position of 0 means the control passes like
the student's wrong text. A position of 1 means it passes like the teacher's text.

## Pre-registered reading

The primary estimate is the pooled hire-reference position in `onpolicy-hire32`. The replication
is the pooled hire-reference position in `onpolicy-full`.

| Reading | Rule, in both rounds | Conclusion |
|---|---|---|
| verdict | position ≥ 0.75 and bootstrap lower bound ≥ 0.50 | The gap comes from the wrong verdict, not from who wrote the text. |
| author | position ≤ 0.25 and bootstrap upper bound ≤ 0.50 | The judge penalises student text. |
| mixed | anything else, or the two rounds disagree | Both contribute. Position is reported as the share of the gap that goes with the verdict, and neither single reading is claimed. |

The reject-reference rows of `onpolicy-full` (chosen 92.6% and rejected 86.0% pooled; these
pooled values are recomputed by `summarize` as `chosen_pass_all` and `rejected_pass_all`) are
secondary and descriptive. The gap there is small, so position is noisy. Per-cell results are
descriptive and are not tested cell by cell.

## Guards

- **Judge drift (retest).** In each round, 400 seeded hire-reference rows have their chosen and
  rejected judged again in the same job. If a side's retest pass rate differs from October's by
  more than 3 pp, or its pass/fail agreement with October is below 90%, then the October
  baseline is not comparable to a judgement made today. The primary reading then uses the
  retest rows (`basis = retest` in `pooled.csv`), where all three sides were judged in the same
  job, and the card says so.
- **Coverage.** If fewer than 70% of a round's hire rows have a control sample, the reading is
  still computed, but it is labelled as limited to the rows the student can get right, and
  `rejected_pass` on matched rows is reported against `rejected_pass_all`.
- **Prompt identity.** Before judging, the job recomputes every October judge prompt from the
  rows and refuses to run if any prompt differs. Same prompt text means the same judge input
  format.

## Text reading

From the October judgements, 20 hire-reference rows in `onpolicy-hire32` whose student
rejection was called `ungrounded`, drawn with seed 42 (`ungrounded_rejections_hire.md`). Each is
read against its CV and job and labelled with one of these:

- (a) **invents.** The rejection states a fact the CV contradicts or does not contain.
- (b) **absence.** The rejection says the CV lacks explicit evidence of something it shows, or
  shows indirectly.
- (c) **judge wrong.** The rejection's claims hold, and the gap it names is real.

The card quotes entries 1 to 5 in drawn order, not five chosen ones. All 20 labels are listed.

## Stop conditions and cost

- A debug job runs first: 64 seeded rows per round, a retest of 16, and separate files under
  `judge_control-debug/`. The full job is submitted only if the debug job ends with
  `STAGE judge-author-control OK`, the control judge parse rate is at least 95%, and the
  prompt-identity check passes.
- The full job fits one debug segment (30 minutes) if the debug job's throughput says it does.
  Otherwise it runs on `normal` with `-t 01:00:00`, after checking the queue. Each judge file is
  cached, so a resubmitted job resumes.
- Cap: 1.5 node-hours in total. No further rounds, prompts or arms without a new approval.

## Splits

Nothing is trained or selected. The benchmark is not read, and the holdout is untouched.
