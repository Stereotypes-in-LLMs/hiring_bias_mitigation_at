# Iterative DPO: what the round-1 model still gets wrong

Iterative DPO refreshes the data between rounds instead of adding epochs. You train on the current pairs, then sample the new model's remaining attribute-induced mistakes as the next round's negatives. This page reports the sampling half of a second round started from the round-1 model π1, what its remaining mistakes look like, and why that round was not trained. The sampler is `generation/onpolicy.py`, the round-2 code is commit `8da7716`, and the outputs are in `artifacts/semisynthetic-v2/onpolicy-it2`. The flip counts come from `scripts/attribute_flips.py`, the rest from the job reports.

## Setup

π1 is r2em checkpoint-100, the SFT student followed by DPO on the error-mass-balanced pairs (β 0.3, plus an NLL term on `chosen` with weight 0.2). On the benchmark it hires 28.4% of candidates and has 2.81% unstable counterfactual sets, against 28.0% and 3.63% for SFT. Checkpoint 100 was picked over the final checkpoint 148 with these audit numbers in view.

The negatives were built exactly like round 1's, with π1 in place of SFT. π1 answered every row of v1's English pool, 35,826 rows over 2,992 CV and job pairs, 8 times at temperature 1 with the attribute and 8 times without. A row is attribute-tipped when π1 is right on most attribute-free samples and wrong on at least one attributed sample. Its first wrong sample that passes the covert filters becomes `rejected`, and the teacher judges both sides. π1 is served merged with its SFT parent. Before `8da7716`, a DPO checkpoint used as the student, or as the start of a training run, was merged without its SFT adapter.

## π1's remaining mistakes point one way

| | SFT (round 1) | π1 |
|---|---|---|
| Rows with a wrong sample | 10,818 | 5,143 |
| Attribute-tipped rows with a wrong sample | 5,614 | 409 |
| Wrong samples in all | 43,036 | 40,422 |
| Samples right with / without the attribute | 84.8% / 85.4% | 85.8% / 86.1% |
| Wrong samples per tipped row, of 8 | 2.2 | 6.9 |
| Strongly tipped rows, and their pairs | 28 on 17 | 293 on 73 |
| Reference of the strongly tipped rows | 27 reject, 1 hire | 288 reject, 5 hire |
| Pairs kept by the judge | 4,267 | 305 |

A row is strongly tipped when it is wrong in at least 6 of 8 samples with the attribute and right in at least 7 of 8 without.

π1 makes about as many wrong samples as SFT, in half as many rows, and most of those rows are wrong without the attribute as well. What is left of the attribute's effect is concentrated and one-directional. On 73 borderline CVs, π1 rejects when no attribute is present and hires as soon as one is. The value hardly matters. "Civilian" flips about as often as "War veteran" (17 and 18 strongly tipped rows), and the tipped rows spread over all three groups (127 to 153 each). Of the 305 pairs the judge kept, 301 have a reject reference.

These are not pairs π1 was trained on. Only 4 of the 73 are in its training data, where its coverage of 582 of the 2,992 pool pairs predicts about 14.

## Sharper, not more dependent on the attribute

The tipped count needs only one wrong sample in eight, so it falls whenever a model samples less noise. Majority decisions avoid that. Counted that way, π1 depends on the attribute less than SFT does, on the pool and on the benchmark.

| | SFT | π1 |
|---|---|---|
| Pool: reject-reference rows whose attributed majority is hire and attribute-free majority is reject | 487 (1.58%) on 139 pairs | 340 (1.11%) on 79 pairs |
| Pool: hire-reference rows flipped the other way | 11 | 5 |
| Benchmark: attributed prompts decided hire where the pair's attribute-free prompt was rejected | 1.34% | 0.50% |
| The same on reject-reference pairs | 0.75% | 0.21% |
| Benchmark: pairs where at least half the attributes turn the reject into hire | 3 | 1 |
| Benchmark: unstable sets at π1's hire rate (op-margin-v2) | 3.77% | 2.81% |

The last row is the matched comparison. π1 minus SFT is −0.96 pp, with a 95% interval from −1.96 to +0.04 (`reports/operating_point/v2/`, job 5063275). The scorer behind it matches HuggingFace at a rank correlation of 0.9997 for SFT and 0.991 for π1.

What changed on the flipped rows is confidence. On π1's 293 strongly tipped rows, SFT hired in 4.5 of 8 attributed samples and was right in only 61% of the attribute-free ones. π1 hires in 7.9 of 8 with the attribute and is right in 99% without. DPO made these decisions confident, so the attribute dependence that is left is crisp instead of noisy. Which of SFT's benchmark decisions π1 changed, prompt by prompt, is in `docs/R2EM_VS_SFT.md`.

On the benchmark, the remaining flips mostly run the other way. For every audited model, an attribute line turns an attribute-free hire into a reject far more often than the reverse.

| Model | Reject to hire, share of prompts | On reject-reference pairs | Pairs flipped by half the attributes | Hire to reject, share of prompts |
|---|---|---|---|---|
| base | 2.83% | 0.81% | 9 | 21.34% |
| SFT | 1.34% | 0.75% | 3 | 5.69% |
| r1-300 | 1.92% | 1.07% | 7 | 4.93% |
| r2-102 | 1.32% | 0.74% | 4 | 2.69% |
| r2em-100 (π1) | 0.50% | 0.21% | 1 | 4.52% |
| r2em-148 | 1.44% | 0.47% | 2 | 2.61% |

## Why round 2 was not trained

The next round's data fails D6 outright. Its share of `chosen` responses that say hire is 0.013, and all 76 validation pairs have a reject reference. Round 1's per-group count balance would keep about 10 of the 305 pairs. A k=32 top-up of the hire rows cannot help when the residual has almost no hire side, so it was not run.

Training on the 229 training pairs as they are would teach "reject" on attributed prompts and nothing else. That is how round 1's r1 run looked better than it was. Its hire rate fell from 28% to 19%, and it gained nothing at matched hire rate. The target would also be a pool behaviour. On the benchmark π1 already has the fewest attribute-to-hire flips of the six models, and its larger remaining flips go the other way. On 2026-10-15 we stopped at this finding.

## Caveats

- 8 samples per prompt at temperature 1. The strongly tipped thresholds are ours, and majority counts leave out 4-to-4 ties.
- `selected.parquet` keeps a row only if one of its wrong samples passed the covert filters. The filters dropped 1% of π1's wrong samples, so a few majority flips may be missing for both models.
- "Wrong" means disagreeing with the v1 teacher's attribute-free reference, a single sample at temperature 0.7. Borderline CVs are where that reference is least certain (open question 2 in `docs/DATA_EVALUATION.md`).
- One π1 and one seed. r1-300 and r2em-148 were not sampled.

## Reproducing

From a Daint login node, one node per submission:

```bash
CONFIG=configs/generation/negatives_v2_onpolicy_it2_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch sample
CONFIG=configs/generation/negatives_v2_onpolicy_it2_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch judge
```

These were jobs 5063103 (21 minutes) and 5063248 (6 minutes). The flip counts need no GPU:

```bash
python scripts/attribute_flips.py \
  --pool SFT=artifacts/semisynthetic-v2/onpolicy-full pi1=artifacts/semisynthetic-v2/onpolicy-it2 \
  --audit SFT=Qwen3.5-9B--en--sft--adapter--en_only_cscs \
          pi1=Qwen3.5-9B--en--dpo--adapter--en_only_v2neg_r2em_step100
```

The configs drafted for training the second round (its data, its DPO run and the hire top-up) are in `8da7716` and were removed afterwards.
