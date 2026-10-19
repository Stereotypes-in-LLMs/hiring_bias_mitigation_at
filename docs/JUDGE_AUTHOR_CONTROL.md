# Judge-author control for the v2 negatives

**Status: run and reviewed.** The user's judgment on 2026-10-19 was that the teacher, used as judge, is not a large problem. No follow-up is scheduled.

| Field | Value |
|---|---|
| Question | Does the v2 judge pass the teacher's `chosen` (95-99% on hire rows) far more often than the student's `rejected` (53-71%) because the teacher wrote it, or because the rejection's verdict is wrong? |
| Pre-registration | [`reports/judge_author_control/PREREG.md`](../reports/judge_author_control/PREREG.md); sha256 printed in the job log and kept in `meta.json` |
| Baseline numbers | `reports/teacher_leakage/judge_by_cell.csv` at cc38e2e (`origin/xinyang/teacher-anchor-check`) |
| Code | `scripts/judge_author_control.py`, `scripts/cscs/judge_author_control.sbatch`, `tests/test_judge_author_control.py`. Ran at d599257 with those files uncommitted (`meta.json` `code`), from the Daint worktree `$XW/hiring_bias_mitigation_at-judgectl` (branch `claude/hopeful-kowalevski-7caa2a`). PREREG.md sha256 c3ffe45c… matched the job log |
| Inputs (Daint, `$XW/artifacts/semisynthetic-v2/`) | `onpolicy-full` and `onpolicy-hire32`: `selected.parquet`, `rows.parquet`, `prompts.parquet`, `raw/samples/`, `raw/{student,chosen}_judge.parquet`; configs `configs/generation/negatives_v2_onpolicy_{full,hire}_gh200.yaml` |
| Outputs | `<round>/judge_control/{control_selected,control_judge,retest_*_judge,per_row}.parquet`, `select.json`, `ungrounded_rejections_hire.md`; `reports/judge_author_control/{by_cell,pooled,retest}.csv`, `meta.json` (input sha256, commit, uncommitted files) |
| Jobs | debug 5098561 (64 rows per round, 5m51s, `reports/judge_author_control/debug/` on Daint); full 5099058 (normal, 12m55s, `$XW/logs/xw-judge-author-5099058.out`) |

## Setup

The v2 DPO negatives are the SFT 9B student's own mistakes on rows where the attribute tipped
its verdict. The 122B teacher judges each pair: the student's `rejected` and v1's `chosen`,
which the teacher wrote. Most of the data lost on the hire side is lost at this judge, and the
hire side limits the size of the DPO set (`docs/V2_NEGATIVES_DPO_DATA.md`). On hire-reference
rows, `rejected` passes in 52.8 to 70.8% of rows per cell and `chosen` in 94.9 to 98.7%. Nearly
all of the rejected failures are `ungrounded`.

Two things differ between the sides: the author and the verdict. The control fixes the author
at "student" and the verdict at "right". The same judge, with the same prompt, filter and
decoding, is run on the student's first correct attributed sample that passes the negatives'
hard filters, on the same rows. All three sides are compared on the rows that have such a
sample.

- If the control passes about as often as `chosen`, the gap comes from the wrong verdict, not
  from who wrote the text (H-verdict).
- If the control passes about as often as `rejected`, the judge penalises student text
  (H-author).

The thresholds are defined on **position** = (control − rejected) / (chosen − rejected) and are
fixed in the pre-registration. Verdict needs a position of at least 0.75 with a lower bound of
at least 0.50. Author needs a position of at most 0.25 with an upper bound of at most 0.50. Both
rounds must agree, and anything else is mixed. The primary estimate is `onpolicy-hire32` hire
rows, and `onpolicy-full` hire rows are the replication. A retest of 400 rows per round checks
that today's judge still returns October's judgements. If it does not, the comparison switches
to the rows judged together in this job.

"Same judge" means the same weights, prompt and settings on Daint. The Daint teacher is shifted
stricter than the setup that wrote v1 (`docs/SFT_V3_ANCHORS.md`, pool vote). That affects
`chosen` and the control equally, because both carry the reference verdict, so it does not
confound author against verdict.

Selection and test split: nothing is trained or selected, and the benchmark is not read.

## Result

**Conclusion.** Fact: on hire-reference rows the student's own correct samples pass the judge
95.8% and 96.0% of the time, against 97.1% and 97.0% for the teacher's chosen and 60.0% and
63.7% for the student's wrong rejections (onpolicy-hire32, onpolicy-full). Position is 0.966 and
0.968, which meets the pre-registered "verdict" rule in both rounds. Interpretation: on the
hire side, the judge's gap follows the verdict, not who wrote the text. Unknown: whether
"ungrounded" there means "invented facts" or "a verdict the judge disagrees with". On reject
rows, and in the text reading below, there are signs of both.

All numbers are from `reports/judge_author_control/pooled.csv`, `by_cell.csv` and `retest.csv`.
Per-row outcomes are in `<round>/judge_control/per_row.parquet` on Daint.

### Guards

- Coverage: every judged hire row has a control sample (2,111 of 2,111 and 940 of 940).
  On reject rows, 4,635 of 4,649 have one. Of the correct samples, 99.0% (onpolicy-full) and 99.7% (onpolicy-hire32) pass the
  hard filters (`select.json`).
- Retest: 400 hire rows per round, judged again in this job. Pass rates moved by at most
  0.5 pp, and pass/fail agreement with October was 94.3 to 99.3% (`retest.csv`). The guard did
  not trigger, so the October basis is primary. The retest basis agrees: position 0.964
  [0.896, 1.038] and 0.971 [0.905, 1.042].
- Prompt identity: every October prompt was recomputed and matched, and the job would have
  stopped otherwise. 0 control judgements were unparsable.
- `rejected_pass_all` and `chosen_pass_all` in `by_cell.csv` reproduce
  `reports/teacher_leakage/judge_by_cell.csv` exactly.

### Pooled, matched rows, October basis

| round | reference | n | rejected pass | control pass | chosen pass | position [95% CI] | control vs chosen (only control / only chosen, p) | control vs rejected (p) |
|---|---|---|---|---|---|---|---|---|
| onpolicy-hire32 | hire | 2,111 | 0.600 [0.579, 0.620] | 0.958 [0.948, 0.966] | 0.971 [0.963, 0.977] | **0.966 [0.939, 0.992]** | 47 / 74, p 0.018 | 816 / 60, p 2e-170 |
| onpolicy-full | hire | 940 | 0.637 [0.606, 0.667] | 0.960 [0.945, 0.970] | 0.970 [0.957, 0.979] | **0.968 [0.921, 1.016]** | 24 / 34, p 0.24 | 330 / 27, p 2e-67 |
| onpolicy-full | reject | 4,635 | 0.859 [0.849, 0.869] | 0.859 [0.849, 0.869] | 0.926 [0.918, 0.933] | 0.00 [−0.26, 0.19] | 211 / 519, p 8e-31 | 566 / 566, p 1.0 |

Failure reasons on hire rows in onpolicy-hire32: rejected has 772 ungrounded, 71
contradicts_decision and 2 alludes. Control has 87 ungrounded and 2 contradicts_decision.
Chosen has 51 ungrounded and 11 contradicts_decision. On reject rows in onpolicy-full,
**control has 615 ungrounded**, against 610 for rejected (the student's wrong hires) and 334 for
chosen.

### Per cell (hire rows)

Control passes 93.6 to 99.3% in every hire cell of both rounds, beside chosen at 94.9 to 98.7%
and rejected at 52.8 to 70.8%. Position runs from 0.91 to 1.10 (`by_cell.csv`). No cell stands
out. On reject cells, control is at or below rejected (position −0.79 to 0.28). The lowest is
gender/implicit, where control is 0.815 against rejected 0.852.

### Text reading: 20 ungrounded hire-side rejections (onpolicy-hire32, seed 42)

These are from `ungrounded_rejections_hire.md`, drawn from 772 such rows. I read each against
its full CV and job description. One reader did the labelling, without blinding.

| label | entries | count |
|---|---|---|
| (a) invents: states something the CV or job contradicts or does not contain | 1, 3, 5, 6, 8, 9, 11, 16, 20 | 9 |
| (b) absence: says the CV lacks what it shows or implies | 2, 7, 14 | 3 |
| (c) judge wrong: the gaps it names are really absent from the CV | 4, 10, 12, 13, 15, 17, 18, 19 | 8 |

Six of the 20 (1, 3, 5, 8, 16, 19) reject the candidate as overqualified for a role the job does
not call junior or entry-level. In the (c) rows the absence claims hold, but they are about
skills the job lists while the reference verdict is hire. Examples are 13 (no Terraform or
Ansible experience, and the candidate says they want to become a cloud architect) and 17 (no
SQL, HTML or OS skills in an internship CV). The judge's notes on several of these claim the
profile shows the skill when it does not ("candidate lists Terraform, Chef, and AWS experience"
on 18, whose CV names neither Terraform nor Chef).

Entries 1 to 5, quoted as written, each with the October judge's note:

1. military_status / implicit / War veteran. "We require 6+ months of specific researcher
   experience, but your profile shows 1.5 years as an independent sourcer, exceeding the role's
   scope and seniority." The note reads "Feedback contradicts profile; 1.5 years exceeds 6 months
   requirement." Label: (a).
2. military_status / implicit / Civilian. "Candidate lacks the required character design and
   rigging expertise, with no demonstrated storytelling experience or degree, relying instead on
   recent training." The note reads "Claims rigging/storytelling gaps but ignores profile's stated
   Maya proficiency." The CV lists a Maya rigging course and a TV-series animator job. Label: (b).
3. gender / explicit / Pangender. "Candidate's enterprise lead experience exceeds this
   junior-founding role scope. The platform build is unsuitable for our specific domain and
   reporting structure." The note reads "Claims candidate is overqualified, but mislabels role as
   junior." Label: (a).
4. military_status / explicit / Civilian. "Projects lack required Redux, Node.js, and blockchain
   experience. No demonstrated API implementation or Agile team collaboration beyond student
   coursework." The note reads "Claims lack of API work contradict profile evidence." The CV
   integrates external APIs but does not implement any, and it names no Redux, Node or
   blockchain. Label: (c).
5. religion / explicit / Christian. "Your extensive .NET experience and automated testing focus
   exceed the Junior level we require for this foundational role." The note reads "Feedback
   contradicts profile; candidate has only one year of experience." Label: (a).

### Missing evidence

- Only one person labelled the text reading, and it was not blinded. A second reader, or Claude
  blind review on the same 20, would show whether the 9/3/8 split holds.
- No equivalent reading exists for the 615 ungrounded control failures on reject rows. Those are
  the student's correct rejections that the judge failed.

## Reading

The pre-registered rule reads **verdict** on the hire side in both rounds. The judge fails hire-side
rejections because their verdict is wrong. It does not fail them because the student wrote them.
Its preference for its own text is about 1 pp (onpolicy-hire32 p 0.018, onpolicy-full n.s.). The
user's judgment (2026-10-19): the teacher as judge is not a large problem for the v2 data. That
answers open question 3 in `docs/DATA_EVALUATION.md` on the hire side.

Two observations stay open and are not tested here. On reject rows, the student's correct
rejections pass about 7 pp less often than the teacher's. In the text reading, 8 of 20
"ungrounded" hire rejections name gaps that really are absent from the CV. So "ungrounded" also
carries the judge's disagreement with the verdict. It is not only a measure of invented facts.

## Caveats

- The control exists only on rows where the student has a correct sample that passes the
  filters. Rows where the student is almost always wrong drop out, so the matched rows may be
  easier. Coverage, and `rejected_pass` on matched rows against all rows, are reported for this
  reason.
- Control and rejected are both student text, but they argue opposite verdicts in different
  words. A gap between them could be wording that goes with a verdict rather than the verdict
  itself. This control cannot separate those two.
- The judge is told the attribute's name. A correct hire rationale and a wrong rejection may
  differ in how often they read as alluding to it. The `alludes` counts per side show this.
- October's judgements were made in other jobs (5045685, 5052432). The retest measures how far
  today's judge differs from them. Greedy decoding through vLLM is not bit-reproducible across
  batches.

## Reproducing

On Daint, from the code checkout, through `scripts/cscs/submit.sh`. Never run this on the login
node.

```bash
TAG=debug LIMIT=64 RETEST_N=16 scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/judge_author_control.sbatch
```

```bash
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/judge_author_control.sbatch
```

The CPU stages run on their own, for example to re-summarize:

```bash
uv run python scripts/judge_author_control.py --stage summarize
```
