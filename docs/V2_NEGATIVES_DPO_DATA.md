# The v2-negatives DPO training sets: round 1 and r2em

Two preference sets built from the v2 negatives reached a DPO run. Round 1 (r1) is the SFT student's own attribute-tipped mistakes over v1's whole English pool, topped up on the hire side and balanced by reference verdict. r2em adds two more pools to round 1's pairs and balances them by count and by the SFT student's error mass. This card says how each set was built, what its data checks showed, and what training on it gave at matched hire rate. The per-item gates and the checks D1 to D7 are defined in `docs/DATA_EVALUATION.md`. The untrained round from π1 is in `docs/ITERATIVE_DPO.md`. `docs/DPO_DATA_CREATION.md` describes the same data stage by stage, from v1's invariant pool, with tables of what the sets contain by attribute.

Every number names its source. Most come from reports on the model drive, which exist only on CSCS Daint under `$HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw/`. Paths below are relative to it, and `v2/` stands for `artifacts/semisynthetic-v2/`. The rest come from commit messages, configs and the JSONs in `reports/operating_point/`.

| Field | Round 1 (r1) | r2em |
|---|---|---|
| Code | sampling `80dd1c9`; hire top-up and count balance `e6b48a5` | builder `2a8d5cd`; Daint run and `reference_decision` `2a272fa`; training config `bf71689` |
| Commit each data job ran at, from the first line of its log in `logs/` | sample 5044997 at `5df925a`; judge 5045685 at `80dd1c9`; top-up sample 5051629, judge 5052432 and check 5053062 at `80dd1c9` | judge 5056706 and score 5056701 at `d56276e`; `candidates` and `build` ran outside Slurm, no log |
| Uncommitted code | sample 5044997 ran `generate_onpolicy_full.py`, first committed in `80dd1c9`; the top-up jobs used `row_filter`, `topup_of` and `balance`, first committed in `e6b48a5`; whether judge 5045685 had local changes is not recorded | the jobs ran the builder, first committed in `2a8d5cd` and `2a272fa` |
| Generation config | `configs/generation/negatives_v2_onpolicy_full_gh200.yaml`, then `negatives_v2_onpolicy_hire_gh200.yaml` | `scripts/build_balanced_dpo.py`; GPU stages through `scripts/cscs/build_balanced_dpo.sbatch` |
| Data config, `data_kind` | `configs/data/en_only_v2.yaml`, `dpo_v2` | `configs/data/en_only_v2_r2em.yaml`, `dpo_round2` |
| Data version | `v2/onpolicy-hire32/balanced/`: `dpo_v2_{train,validation}.parquet`, `balance.json`, `direction_report.{md,json}` | `v2/round2-error-mass/`: `dpo_round2_{train,validation}.parquet`, `dpo_round2_report.json`, `dpo_round2_candidates_report.json` |
| Upstream data | `artifacts/semisynthetic-v1` (pool and `chosen` side); SFT student `Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en`, merged at `outputs/merged/qwen3.5-9b-hiring-debias-sft-en` | round 1's pairs, `v2/round2-error-mass/onpolicy_reject_surplus.parquet`, the `semisynthetic-v1` DPO pairs |
| Training run | `configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1_gh200.yaml` to `outputs/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1`; jobs 5054142 (`ead07d5`), 5054909 and 5055259 (`df41b19`); W&B name `dpo-qwen3.5-9b_en_only_dpo_v2neg_r1` | `qwen3.5-9b_en_only_dpo_v2neg_r2em_gh200.yaml` to `outputs/dpo/qwen3.5-9b_en_only_dpo_v2neg_r2em`; jobs 5056917 and 5056952 (`0d4f59d`); W&B name `dpo-qwen3.5-9b_en_only_dpo_v2neg_r2em` |
| Evaluation runs | `Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r1_step{100,300}`, audit job 5055675 | `Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step{50,100,148}`, audit job 5057153 |
| Matched comparison | `reports/operating_point/<run>--vs--Qwen3.5-9B--en--sft--adapter--en_only_cscs.json` | the same; step 100 also under `reports/operating_point/v2/` |
| Per-sample data | the parquets above, `samples.md` beside each judge output, `direction_student_scores.parquet` for D4 | the parquets above, `dpo_round2_judge_sample.csv` for the judge, `dpo_round2_scores.parquet` for the SFT margins |

## Setup

### Round 1

**Sampling over the pool.** `scripts/generate_onpolicy_full.py` with `negatives_v2_onpolicy_full_gh200.yaml` takes every row of v1's English pool. A row is one CV and job pair with one attribute in one condition. The merged SFT student answers each row's attributed prompt 8 times at temperature 1.0, and each pair's attribute-free prompt 8 times. A row is attribute-tipped when more than half of its attribute-free samples match the reference verdict (`min_free_right: 0.5`). Its first wrong attributed sample that passes the hard filters becomes `rejected`, and `chosen` stays v1's teacher text (`generation/onpolicy.py::select_negatives`).

**Judge and split.** The 122B teacher judges each selected negative and its `chosen` side on the batch-01 rubric, and a failing side drops the pair. The split is by candidate, and v1's held-out candidates stay held out. Output: `v2/onpolicy-full`.

**Hire top-up.** v1's English pool is mostly reject, so the pairs were too. `negatives_v2_onpolicy_hire_gh200.yaml` samples only the hire-reference rows again (`row_filter: {reference_decision: hire}`), at 32 samples per prompt instead of 8, with everything else unchanged. Its split follows the base run's rows (`topup_of`), so a candidate sits on the same side in both runs. Output: `v2/onpolicy-hire32`.

**Merge and balance.** The top-up run's `balance` stage merges the base run's pairs with its own, inside each split. When both runs kept a pair for the same row, the base run's pair stays (`onpolicy.merge_topup`). `onpolicy.balance_by_reference` then keeps, within each protected group, every pair of the scarcer reference verdict and a seeded sample of the same size from the other. Output: `v2/onpolicy-hire32/balanced/`, which `en_only_v2.yaml` points at.

**Training.** β 0.1, an NLL term on `chosen` with weight 0.2, learning rate 5e-5, 300 steps at an effective batch of 16 (config). The last segment ended at epoch 2.098 (log of job 5055259). Checkpoint 100 had the lowest eval loss and the highest decision agreement, 0.962 (header of `configs/audit/qwen3.5-9b_en_dpo_v2neg_r1_step100_gh200.yaml`). Checkpoint 300 is the last.

### r2em

Round 1 became stricter without becoming more invariant (Result below). Commit `65bb6b2` traced that to the D4 asymmetry and tried a margin-balanced round 2 (r2, `configs/data/en_only_v2_r2.yaml`), which this card does not describe. r2em keeps r2's training settings and changes only the data (`bf71689`).

**Pools.** `scripts/build_balanced_dpo.py candidates` reads three kinds of pool into one schema (`generation/balanced_dpo.py`). Each pair keeps its source, and the source decides which pair wins a conflict.

| Source | What it is | Wins a conflict |
|---|---|---|
| `round1` | round 1's train and validation pairs | first |
| `extra:onpolicy_reject_surplus` | per `en_only_v2_r2em.yaml`, the reject-side pairs round 1's count balance dropped; training only | second |
| `legacy` | the `semisynthetic-v1` DPO pairs whose `rejected` does not name the attribute, re-checked with the current detector (`filter_legacy`) | third |

**Judge.** The lexical detector misses paraphrase. The 122B teacher therefore reads both rationales of every legacy pair and answers whether either refers to the protected characteristic. A pair stays only if both answers parse as "no" (`balanced_dpo.parse_judge`).

**Score.** The SFT student scores every prompt still in play with HF and PEFT. The score is the decision margin, logit(hire) minus logit(reject) at the decision token (`eval/margins.py`). From it, `p_wrong` is the student's probability of the `rejected` decision against the `chosen` one. Summed over one side, `p_wrong` is that side's error mass, the expected number of SFT errors.

**Build.** `balanced_dpo.assemble` applies, in order:

1. the judge filter;
2. one pair per prompt, the higher-priority pool's pair kept;
3. one decision per CV and job. The highest-priority pool holding a pair sets its decision, and a lower pool's pair that disagrees is dropped;
4. the candidate split. A candidate that any pool held out never trains;
5. `assert_no_leakage` against the benchmark holdout, on both splits;
6. `balance_error_mass` on the training split only. The heavier side drops its most confidently wrong pairs until its error mass is within 5% of the lighter side's. The lighter side keeps its hardest pairs. Both sides are then cut to the same count by dropping their easiest pairs.

Validation is not balanced, because it never reaches the gradient.

**Training.** β 0.3, NLL weight 0.2, learning rate 5e-5, one epoch of 148 steps at an effective batch of 16, and a greedy decision eval on 64 validation prompts after every eval (config, `bf71689`). Checkpoints 50, 100 and 148 were audited (`0d4f59d`).

## Result

### Round 1: the full-pool run

| Step | Number | Source |
|---|---|---|
| Pool | v1's invariant filter keeps 35,826 of 36,000 rows; 2,992 CV and job pairs; 38,818 prompts | `v2/onpolicy-full/prepare.json` |
| Student right, with / without the attribute | 84.8% / 85.4% of samples | `v2/onpolicy-full/sample_report.json` |
| Rows tipped; with a wrong sample; both | 30,585; 10,818; 5,614 | same |
| Wrong samples; naming the attribute; naming its group | 43,036; 0.10%; 0.12% | same |
| Wrong samples through the hard filters | 42,644; largest drop `negative_length_mismatch`, 249 | same |
| Selected negatives, tipped / untipped | 5,589 / 5,201; only tipped ones go on | same |
| Teacher judge on `rejected` | keeps 4,595 of 5,589; drops 923 ungrounded, 69 contradicting their decision, 2 alluding | `v2/onpolicy-full/report.json` |
| Teacher judge on `chosen` | keeps 4,267 of 4,595; drops 316 ungrounded, 12 contradicting | same |
| Split | 3,615 train / 652 validation; 359 validation candidates | same |
| Judge plausibility, mean of 1 to 5 | 4.25; hire-reference 3.45, reject-reference 4.42 | same |
| `chosen` hire share (D6) | 0.137, FAIL | same, and `v2/onpolicy-full/direction_report.md` |
| D3 balanced AUC | 0.650 on 1,168 balanced pairs, FAIL against ≤ 0.65 | `v2/onpolicy-full/direction_report.md` |
| D1, D2, D5 | pass; lexical 0.000, judge allusion 0.000; mean Jaccard 0.125 | same |

### Round 1: the hire top-up

| Step | Number | Source |
|---|---|---|
| Rows | 5,072 hire-reference rows on 424 pairs; 5,496 prompts; 32 samples each | `v2/onpolicy-hire32/prepare.json` |
| Student right, with / without the attribute | 93.8% / 92.9% | `v2/onpolicy-hire32/sample_report.json` |
| Rows with a wrong sample; of them tipped | 2,324; 2,120 | same |
| Wrong samples | 9,888 | same |
| Tipped negatives selected | 2,111, against 940 at k=8 | `v2/onpolicy-hire32/report.json`; the 940 is from `e6b48a5` |
| Teacher judge on `rejected` | keeps 1,266 of 2,111; drops 772 ungrounded, 71 contradicting, 2 alluding | `v2/onpolicy-hire32/report.json` |
| Teacher judge on `chosen` | keeps 1,224 of 1,266 | same |
| Judge plausibility, mean | 3.39 | same |
| Split | 1,092 train / 132 validation | same |

### Round 1: merge, balance and the final checks

From `v2/onpolicy-hire32/balanced/balance.json`:

| | Train | Validation |
|---|---|---|
| Base run's pairs | 3,615 | 652 |
| Top-up pairs; new rows; rows the base run already had | 1,092; 622; 470 | 132; 79; 53 |
| Merged | 4,237 | 731 |
| Kept per side: gender, military status, religion | 339, 391, 409 | 44, 59, 43 |
| Dropped: gender, military status, religion | 632, 679, 648 | 140, 148, 151 |
| Kept | 2,278 | 292 |

The final set has 2,570 pairs and a `chosen` hire share of 0.50. Its checks, from `v2/onpolicy-hire32/balanced/direction_report.md`:

| Check | Result | Threshold | Verdict |
|---|---|---|---|
| D1 validity | 0 `chosen` off the reference, 0 `rejected` not flipped | 0 and 0 | pass |
| D2 covertness | lexical 0.000, judge allusion 0.000 | ≤ 0.05, ≤ 0.1 | pass |
| D3 balanced AUC | 0.684 on 2,570 pairs; no interval recorded | ≤ 0.65 | **fail** |
| D5 mean Jaccard | 0.126 | flag above 0.8 | ok |
| D6 hire share | 0.500 | 0.4 to 0.6 | pass |

D4 ran on a 600-pair subsample with HF and PEFT. "Prefers `chosen`" compares whole-sequence log-probabilities. "Decides the reference" is the sign of the decision-token margin.

| Slice | n | SFT prefers `chosen` | SFT decides the reference | Base prefers `chosen` | Base decides the reference |
|---|---|---|---|---|---|
| all | 600 | 0.620 | 0.930 | 0.615 | 0.640 |
| hire reference | 291 | 0.818 | 0.979 | 0.598 | 0.392 |
| reject reference | 309 | **0.434**, below the 0.5 to 0.9 band | 0.883 | 0.631 | 0.874 |

### r2em: the pools

From `v2/round2-error-mass/dpo_round2_candidates_report.json`:

| Source | Split | Hire | Reject |
|---|---|---|---|
| `round1` | train | 1,139 | 1,139 |
| `round1` | validation | 146 | 146 |
| `extra:onpolicy_reject_surplus` | train | 0 | 1,959 |
| `legacy` | train | 1,669 | 0 |
| `legacy` | validation | 89 | 0 |

`filter_legacy` took v1's 31,476 training pairs, kept the 18,935 in English, and dropped 17,266 whose `rejected` names the attribute. On validation it went from 1,647 to 976 to 89. The detector re-check dropped none.

### r2em: the build

From `v2/round2-error-mass/dpo_round2_report.json`:

| Stage | Number |
|---|---|
| Judge | 1,758 legacy pairs judged; 1,698 dropped as referring or unreadable |
| Duplicate prompts | 11 dropped, all legacy |
| Contradicting anchors | 0 dropped |
| Training rows of held-out candidates | 4 dropped, all legacy |
| Trimmed as most confidently wrong | 1,771 reject pairs, each with `p_wrong` ≥ 0.1645: 1,119 from the surplus, 652 from round 1 |
| Dropped as easiest, for equal counts | 146 reject, 0 hire |
| Error-mass ratio after | 1.0496 |
| Steps per epoch at batch 16 | 148 |
| Parameters | tolerance 0.05, `min_keep` 0.5, seed 42; scored by `outputs/sft/qwen3.5-9b-hiring-debias-sft-en` |

| Side | Pairs | Error mass | Mean `p_wrong` | SFT wrong | Sources |
|---|---|---|---|---|---|
| hire, before = after | 1,181 | 106.41 | 0.0901 | 13 | round1 1,139, legacy 42 |
| reject, before | 3,098 | 758.15 | 0.2447 | 318 | surplus 1,959, round1 1,139 |
| reject, after | 1,181 | 111.69 | 0.0946 | 0 | surplus 748, round1 433 |
| validation hire | 149 | 15.63 | 0.1049 | 0 | round1 146, legacy 3 |
| validation reject | 146 | 40.82 | 0.2796 | 22 | round1 146 |

The final set has 2,362 training pairs and 295 validation pairs. No direction check on it is recorded: `v2/round2-error-mass/` holds no `direction_report`.

### What training on each set gave

Each run is compared with the SFT student audited on the same GH200 stack, `Qwen3.5-9B--en--sft--adapter--en_only_cscs`, on 2,700 single-group sets and 31,050 prompts. SFT hires 27.3% and has 3.63% unstable sets. The SFT arm is matched to each run's hire rate. Each row comes from one JSON, `reports/operating_point/<run>--vs--Qwen3.5-9B--en--sft--adapter--en_only_cscs.json`. r2 is there for the data-only comparison in Reading.

| Run | Hire rate | Unstable sets | SFT at that hire rate | Δ matched, pp (95% CI) |
|---|---|---|---|---|
| r1 step 100 | 19.0% | 3.30% | 2.71% | +0.58 (−0.50, +1.64) |
| r1 step 300 | 19.1% | 2.74% | 2.78% | −0.04 (−0.90, +0.84) |
| r2em step 50 | 21.3% | 3.44% | 3.61% | −0.17 (−1.39, +0.99) |
| r2em step 100 | 28.4% | 2.81% | 3.77% | −0.96 (−1.96, +0.04) |
| r2em step 148 | 30.0% | 3.37% | 4.07% | −0.70 (−1.74, +0.39) |
| r2 step 102, margin-balanced | 20.1% | 2.63% | 2.90% | −0.27 (−1.22, +0.70) |

None of the intervals excludes zero (`c240480`). The r2em step-100 figure is the same under the op-margin-v2 standard, whose JSON in `reports/operating_point/v2/` conforms (job 5063275, `8c4afb6`). Round 1 also lost 3.6 pp of utility (`65bb6b2`).

## Reading

**Round 1 balanced counts, not pressure.** Fact. The final set is 50% hire. On its hire-reference pairs SFT prefers `chosen` 81.8% of the time, and on its reject-reference pairs 43.4%. Fact. Both round-1 checkpoints hire about 19% of benchmark candidates against SFT's 27.3%, and neither beats SFT at matched hire rate. Interpretation. The reject-reference pairs keep a large DPO gradient after the hire-side pairs go quiet, so the net push is toward reject. `65bb6b2` and the docstring of `generation/balanced_dpo.py` make this argument. It fits the hire-rate drop. Nothing in this card tests it directly.

**The asymmetry sits in the rationale, not the decision token.** Fact. On reject-reference pairs SFT's decision token already favours the reference on 88.3%, yet its whole-sequence likelihood favours `chosen` on only 43.4%. Interpretation. There `rejected` is the student's own sampled text and `chosen` is the teacher's. The student finds its own wording more likely even where it gets the decision right. Not confirmed. `direction_student_scores.parquet` holds the per-token margins that would show it.

**r2em balances the decision token, and drops most of the errors to do it.** Fact. The reject side started with an error mass of 758.15 against the hire side's 106.41. To close that gap, the build trimmed the 1,771 reject pairs with the highest `p_wrong`, and those included all 318 reject pairs SFT gets wrong. Of round 1's 1,139 reject training pairs, 433 remain. Fact. After the balance, mean `p_wrong` is 0.090 on the hire side and 0.095 on the reject side, and SFT gets 13 of the 2,362 decisions wrong. Interpretation. r2em is a set SFT already decides almost entirely right, so its DPO signal is mostly on confidence and rationale tokens. Its later checkpoints hire at about SFT's rate, 28.4% and 30.0%, which fits that reading. Step 50 hires 21.3%, so the hire rate dipped before it recovered. The audits cannot say why. The balance also works on the decision-token margin, while the D4 asymmetry above is in the whole sequence. Whether r2em equalises what DPO actually optimises is not measured.

**The legacy pool added almost nothing.** Fact. Of v1's 18,935 English training pairs, the lexical filter let 1,669 through, and the judge then flagged nearly all legacy pairs (1,698 of 1,758). 42 legacy pairs reached training, all on the hire side. `2a272fa` quotes the kind of wording the judge caught, "your religious holidays" and "we prefer a male candidate". Interpretation. v1's negatives are overt, and the lexical detector does not see paraphrase. Any reuse of v1 pairs needs the judge.

**What these results do not separate.** r1 and r2em differ in data, in β (0.1 against 0.3) and in length (about 2.1 epochs against one). r2 and r2em share every training setting, so those two isolate the data. r2 step 102 hires 20.1%, and r2em's later checkpoints hire 28.4% and 30.0%. Neither differs from SFT at matched hire rate. The checkpoints were not paired by a rule set in advance, so this describes the runs and does not test the balance.

## Caveats

- **D3 fails on both round-1 sets.** The full-pool pairs score 0.650 on 1,168 balanced pairs and the final set 0.684 on 2,570, against a limit of 0.65. No interval is recorded for either. DPO trained on the final set anyway.
- **D4 is below its band on the reject side of round 1.** 0.434 on 309 pairs. The overall 0.620 passes only because the hire side is high.
- **No direction check on r2em's final set.** Two properties hold by construction on its training split. Every pair flips, since `normalise_pairs` drops pairs whose sides decide alike, and the two sides have equal counts. D3 and D4 on the r2em set are unknown.
- **How the surplus file was written is not recorded.** `onpolicy_reject_surplus.parquet` has 1,959 reject training pairs. That equals the training pairs round 1's balance dropped, 632, 679 and 648 per group in `balance.json`, all on the reject side because hire was the scarcer verdict. The `candidates` and `build` stages ran outside Slurm, per the header of `build_balanced_dpo.sbatch`, so no log holds their command lines.
- **Most data jobs ran on uncommitted code.** The commit in their log header predates the commit that adds the code they used (code table above). The diffs are not recorded.
- **The reference verdict is one teacher sample at temperature 0.7.** "Wrong" means disagreeing with it, and the borderline CVs where tipping happens are where it is least certain (open question 2 in `docs/DATA_EVALUATION.md`). The pool's 86% reject rate comes from this reference (`e6b48a5`).
- **The teacher judges `chosen` texts it wrote itself**, in both sets.
- **Checkpoint selection.** r1 step 100 was chosen on validation eval loss, and step 300 is the last. r2em step 100 was later picked as π1 with these audit numbers in view (`docs/ITERATIVE_DPO.md`). This card reports every audited checkpoint and selects none.
- **The op-margin v1 JSONs say `floor_check: unverified`.** Only r2em step 100 has a conforming op-margin-v2 report. The r2em-148 comparison against r2-102 was left out for that reason (`c240480`).
- **The commit message of `80dd1c9` gives 3,000 pairs and 38,826 prompts.** `prepare.json` records 2,992 and 38,818, and this card uses those.
- **`onpolicy-full/balanced/` is not on disk**, although `en_only_v2.yaml` mentions it. Its 1,168 pairs match D3's balanced sample of the full-pool run.
- One seed throughout. 8 samples per prompt on the base run and 32 on the top-up, at temperature 1.

## Reproducing

From a Daint login node, one node per submission, one job at a time. Round 1:

```bash
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch sample
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch judge
CONFIG=configs/generation/negatives_v2_onpolicy_hire_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch sample
CONFIG=configs/generation/negatives_v2_onpolicy_hire_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch judge
CONFIG=configs/generation/negatives_v2_onpolicy_hire_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch check
```

The last `check` merges the base run's pairs, balances, and runs D1 to D6 and D4 on `v2/onpolicy-hire32/balanced/`.

r2em has two CPU stages and two GPU stages. Run the CPU stages inside a job, not on the login node. The inputs are the ones `dpo_round2_candidates_report.json` names. The round-1 pairs are `v2/onpolicy-hire32/balanced/dpo_v2_{train,validation}.parquet`, whose sizes match the report. Where v1's DPO pairs were read from is not recorded.

```bash
python scripts/build_balanced_dpo.py candidates --artifacts-dir artifacts/semisynthetic-v2/round2-error-mass \
  --round1-train artifacts/semisynthetic-v2/onpolicy-hire32/balanced/dpo_v2_train.parquet \
  --round1-validation artifacts/semisynthetic-v2/onpolicy-hire32/balanced/dpo_v2_validation.parquet \
  --extra artifacts/semisynthetic-v2/round2-error-mass/onpolicy_reject_surplus.parquet \
  --legacy-dir <folder with v1's dpo_train/dpo_validation.parquet>
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/build_balanced_dpo.sbatch judge
scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/build_balanced_dpo.sbatch score
python scripts/build_balanced_dpo.py build --artifacts-dir artifacts/semisynthetic-v2/round2-error-mass
```

`build` uses the defaults `--tolerance 0.05 --min-keep 0.5 --seed 42`, which `dpo_round2_report.json` records under `parameters`.

Training, on debug in 30-minute segments that resume:

```bash
CONFIG=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/train_dpo.sbatch
CONFIG=configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r2em_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/train_dpo.sbatch
```

The construction is pinned by tests that need no GPU. `tests/test_onpolicy.py` covers selection, the top-up merge and the count balance. `tests/test_balanced_dpo.py` covers the pools, the judge parser, the anchors, the candidate split and the error-mass balance.
