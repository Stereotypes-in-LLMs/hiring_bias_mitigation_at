# Rejection-sampling SFT: the optimizer control

DPO and GRPO continued from the SFT model each gained about 1 pp of counterfactual consistency at matched hire rate (`docs/ITERATIVE_DPO.md`, `docs/GRPO_PILOT.md`). This control keeps their anchors and replaces the objective with the plainest one: a second SFT stage on the student's own right answers, as in ReST or rejection-sampling fine-tuning. It needs no generation, because the student's samples were already on disk. The data builder is `scripts/build_rft_data.py`, the trainer `mitigation/sft.py` from `sft_checkpoint`, and the branch `xinyang/grpo-pilot`. The run is job 5066776, the audit 5066817, and the matched comparison 5066868.

The short version: it does not move the number. At matched hire rate the second stage is +0.31 pp against SFT (95% CI −0.39 to +1.00). It also made the model stricter, as every other arm trained on these anchors did except r2em. Its hire rate fell from 28.0% to 23.7%, although every one of its hire-side training rows was a case where the student wrongly rejects.

## Setup

| | |
|---|---|
| Model | Qwen3.5-9B with the published SFT adapter merged in, plus a fresh LoRA (r 32, α 64, dropout 0.05) |
| Data | `rft-01`: 2,400 attributed English pool rows, 1,200 per reference verdict; the same 119 held-out candidates as the GRPO pilot |
| Target | the row's first stored student sample (the k=32 hire top-up first, then the k=8 pool) that reaches the attribute-free reference and passes `filter_invariant` and the group filter |
| Balance | equal error mass per half, as r2em: 117.2 against 117.0 |
| Optimisation | lr 5e-5, one epoch, 16 rows per step: 150 steps, 17 minutes on one GH200 node |

34,226 of the pool's 35,826 rows have such a target. The 1,599 reject rows without one are rows where the student hires in every sample, so rejection sampling cannot train on its most one-sided mistakes. Hire rows carry little error each (the student rejects in about one sample of ten). The hire half therefore takes all 1,200 of its tipped rows, and the reject half matches that mass with 74 tipped rows and 1,126 the student already gets right.

The update is the size of DPO r2em's (mean |ΔW| 2.4e-5), and a bf16 merge keeps 90% of it. The audited model is close to the trained one here, unlike the GRPO pilot's (58%). The scorer reproduces HuggingFace on the merged weights at a decision agreement of 1.000 and a rank correlation of 0.9997.

## Result

| | SFT | r2em-100 | GRPO pilot | RFT |
|---|---|---|---|---|
| Benchmark hire rate | 28.0% | 28.4% | 19.9% | 23.7% |
| Hires on hire-reference prompts | 65.8% | 66.5% | 50.6% | 56.9% |
| Hires on reject-reference prompts | 9.1% | 9.3% | 4.6% | 7.1% |
| Agreement with the reference | 82.5% | 82.6% | 80.4% | 80.9% |
| Unstable sets at its own hire rate | 3.63% | 2.81% | 1.93% | 3.93% |
| SFT moved to the same hire rate | – | 3.77% | 2.87% | 3.62% |
| Δ against SFT at matched hire rate | – | −0.96 pp | −0.95 pp | **+0.31 pp** |
| 95% CI | – | −1.96 to +0.04 | −1.90 to −0.05 | **−0.39 to +1.00** |

All three reports conform (`reports/operating_point/v2/`).

The changes run one way here too. On identical prompts, 4.3% of SFT's decisions went from hire to reject and 0.01% the other way. On the attribute-free prompts, 22 of SFT's 129 hires became rejects and none went the other way. Counted at its own operating point, the attribute's effect hardly changed (`scripts/attribute_flips.py`):

| Model | Attribute-free hires / rejects | Reject to hire, share of prompts | Hire to reject, share of prompts |
|---|---|---|---|
| SFT | 129 / 321 | 1.34% | 5.69% |
| GRPO pilot | 89 / 361 | 0.90% | 2.78% |
| RFT | 107 / 343 | 1.37% | 4.76% |

## Reading

The control separates two things the arms share from one they do not.

What they share is the direction. Imitating the student's own right answers on these anchors, with the hire half made entirely of rows the student wrongly rejects, still lowered the hire rate by 4 pp. Round-1 and round-2 DPO and the GRPO pilot lowered it by 8 to 9 pp. The anchors are the v1 teacher's attribute-free verdicts. They reject 85.8% of the pool's 2,992 pairs, while the benchmark's GPT-4o reference rejects 66.7% of its 450. The two sets are different CVs, so this alone does not prove a stricter standard rather than a weaker candidate pool. But every arm that learned more of the anchors' decisions on borderline CVs moved toward rejecting borderline benchmark candidates. r2em is the exception. One difference that may explain it is that r2em's NLL term was on the teacher's invariant `chosen` text, not on the student's own answers; this run cannot test that.

What they do not share is the gain. The objectives that push away from the student's attribute-dependent answers (DPO's rejected side, GRPO's negative advantages) reached −0.95 to −0.96 pp at matched hire rate. Plain imitation reached +0.31 pp, inside its noise. So the optimizer is not irrelevant: about 1 pp comes from contrast. That 1 pp is the size of the bootstrap interval's half-width, and no arm has exceeded it.

So the next lever is the anchor: majority-vote references with self-disagreeing pairs dropped, or a reference whose base rate matches the benchmark's. Another optimizer on the same anchors is not it.

## Caveats

- One run and one seed. The decision eval during training (64 held-out rows with a right sample by construction) read 1.000 from step 50 on and says nothing about the benchmark.
- Equal error mass leaves the reject half mostly easy rows. A count-balanced draw with more tipped reject rows would push harder toward reject, by the GRPO smoke run's evidence.
- The matched comparison sits at 23.0%, between SFT's and GRPO's operating points.

## Reproducing

From a Daint login node, one submission at a time, with the worktree as `REPO` and its `src` first on `PYTHONPATH`:

```bash
export REPO=$PWD PYTHONPATH=$PWD/src
python scripts/build_rft_data.py                 # CPU, about a minute: artifacts/semisynthetic-v2/rft-01
CONFIG=configs/mitigation/sft/qwen3.5-9b_en_only_rft_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/train_sft.sbatch
CONFIGS=configs/audit/qwen3.5-9b_en_rft_gh200.yaml \
  scripts/cscs/submit.sh -p normal -t 01:00:00 scripts/cscs/run_audits.sbatch
VALIDATE=configs/audit/qwen3.5-9b_en_rft_gh200.yaml:adapter \
PASSES=configs/audit/qwen3.5-9b_en_rft_gh200.yaml:adapter \
COMPARE=configs/audit/qwen3.5-9b_en_rft_gh200.yaml:configs/audit/qwen3.5-9b_en_sft_en_only_gh200.yaml \
  scripts/cscs/submit.sh -p normal -t 01:00:00 scripts/cscs/op_control.sbatch
```

These were jobs 5066776 (17 minutes), 5066817 (18 minutes) and 5066868 (20 minutes). Job 5066765 died at trainer construction on `loss_type=None`, which TRL 1.15 rejects. `sft.py` now leaves unset fields to TRL.
