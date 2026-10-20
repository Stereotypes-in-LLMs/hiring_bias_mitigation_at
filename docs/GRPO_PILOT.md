# GRPO from SFT: a pilot

GRPO replaces DPO's teacher-written or student-sampled `rejected` text with a reward. The policy samples eight answers to each attributed prompt, and each answer is rewarded for reaching the pair's attribute-free reference verdict. A wrong answer costs the same in either direction, and no negatives have to be found. This page reports one pilot run from the SFT model, what it did to the benchmark, and why. The trainer is `mitigation/grpo.py`, the prompt set `scripts/build_grpo_prompts.py`, and the branch `xinyang/grpo-pilot`. The run is job 5066266, the audit 5066531, and the matched comparison 5066563.

The short version: at its own hire rate GRPO has fewer unstable counterfactual sets than SFT, −0.95 pp (95% CI −1.90 to −0.05), the same size as DPO's best round (r2em-100, −0.96 pp). It got there the way round-1 DPO did. Every decision it changed went from hire to reject, and the benchmark hire rate fell from 28.0% to 19.9%. Its new rejections rest mostly on requirements read more strictly than the job states them, not on invented facts.

## Setup

| | |
|---|---|
| Policy | Qwen3.5-9B with the published SFT adapter merged in, plus a fresh LoRA (r 32, α 64, all projections, no dropout) |
| Reference for the KL term | the same model with the LoRA disabled, i.e. SFT; β 0.04 |
| Prompts | `grpo-02`: 3,000 attributed English pool prompts, 1,500 per reference verdict; 119 candidates of earlier validation splits held out |
| Reward | +1.0 when the parsed decision equals the attribute-free reference; +0.2 when the answer is exactly the JSON object with a 4-to-45-word feedback |
| Rollouts | 8 per prompt at temperature 1, at most 96 tokens, TRL 1.15 GRPOTrainer with one colocated vLLM 0.31 engine per GPU |
| Optimisation | lr 1e-5 constant after 5 warm-up steps, 16 prompts (128 completions) per step, one epoch: 187 steps, 74 minutes on one GH200 node |

The SFT student's own samples (8 per row, 32 for the hire top-up) give each pool row its error probability p. The prompt set draws exactly half its rows per verdict. Within each half, the share of attribute-tipped rows is set so that both halves carry the same summed sqrt(p(1−p)). That is the expected pull on the decision under group-normalised advantages: 390.7 on each side. All 1,500 hire rows are tipped, and 873 of the reject rows. An earlier set with half of each side tipped (`grpo-01`) gave the reject half 1.5 times the pull, and a 20-step smoke run on it (job 5065891) moved only that side. The sampled hire rate on reject-reference prompts fell from 29% to about 5%, while on hire-reference prompts it did not move.

Three things keep the training honest about what it serves:

- **Weight sync.** vLLM receives merged weights only. They are computed out of place in fp32 and cast once to bf16, under the checkpoint's `model.language_model.` names, so vLLM's LoRA path (`mitigation/merge.py`) is never used and the frozen base is never written.
- **Sync check.** At the start and every 50 steps, every engine's margins on 256 held-out prompts must match HuggingFace running the same merged weights, with `operating_point_control validate`'s thresholds. They matched every time, at 99.6–100% decision agreement and rank correlation 0.9995–0.9997.
- **Decision monitor.** Every 25 steps, greedy decisions on 256 held-out rows, through HF + PEFT and through the served engine. The run stops if the projected benchmark hire rate leaves 20–40%. It never stopped; "Why the monitor did not stop it" below says why.

## Result

| | SFT | r1-300 | r2em-100 | GRPO |
|---|---|---|---|---|
| Benchmark hire rate | 28.0% | 19.1% | 28.4% | 19.9% |
| Hires on hire-reference prompts | 65.8% | 47.6% | 66.5% | 50.6% |
| Hires on reject-reference prompts | 9.1% | 4.8% | 9.3% | 4.6% |
| Agreement with the reference | 82.5% | 79.3% | 82.6% | 80.4% |
| Unstable sets at its own hire rate (op-margin-v2) | 3.63% | – | 2.81% | 1.93% |
| SFT moved to the same hire rate | – | – | 3.77% | 2.87% |
| Δ against SFT at matched hire rate | – | −0.04 pp (v1) | −0.96 pp | **−0.95 pp** |
| 95% CI | – | −0.90 to +0.84 | −1.96 to +0.04 | **−1.90 to −0.05** |

Both version-2 reports conform (`reports/operating_point/v2/`). The scorer reproduces HuggingFace on the GRPO checkpoint's merged weights at a decision agreement of 1.000 and a rank correlation of 0.998, and the audit's greedy decisions at 1.000. r1-300's row comes from the version-1 control (top-20 log-probs), so it is not directly comparable.

### What changed runs one way

On identical prompts, GRPO turned 8.1% of SFT's decisions from hire into reject and none from reject into hire. On the attribute-free prompts that is 40 of SFT's 129 hires, and none the other way. A hire was lost on 15.3% of hire-reference prompts and on 4.5% of reject-reference prompts. The model became stricter, mostly where the reference says hire.

Counted at each model's own operating point (`scripts/attribute_flips.py`), the attribute's effect shrank:

| Model | Attribute-free hires / rejects | Reject to hire, share of prompts | Hire to reject, share of prompts |
|---|---|---|---|
| SFT | 129 / 321 | 1.34% | 5.69% |
| r1-300 | 83 / 367 | 1.92% | 4.93% |
| r2em-100 | 132 / 318 | 0.50% | 4.52% |
| GRPO | 89 / 361 | 0.90% | 2.78% |

Part of that is the operating point. A model that hires a third fewer people has a third fewer attribute-free hires to flip, so the matched comparison above is the number to cite. Matched, the gain is the same −0.95 pp that DPO reached.

### Why the hire rate fell although the halves were balanced

The pull was balanced at SFT's starting rates and did not stay balanced. Hire-reference prompts saturated first. Their sampled hire rate rose from 0.90 to 0.95 within 50 steps and to 0.99 by the end. Over the run, the share of prompt groups whose eight answers all scored the same rose from 64% (steps 1–25) to 95% (steps 151–175). Those groups carry no gradient. What signal was left came mostly from reject-reference prompts, where the policy kept sampling a few hires (10% falling to 6%). From about step 50 on, the run was mostly teaching "reject".

That lesson lands on the benchmark's borderline candidates. The pool's hire-reference rows are easy: SFT samples hire on 90% of them and decides hire greedily on all 128 monitor rows. The benchmark's hire-reference prompts are harder, and SFT hires only 65.8% of them. A general shift toward reject flips few decisions on the first kind and many on the second.

### Why the monitor did not stop it

The monitor's hire-side rows sat at SFT's 100%. Over the run they fell only to 92–98%, so their ratio-based projection read 0.235–0.270 at every check while the benchmark went to 0.199. A decision count on saturated rows cannot see a threshold that moves. The monitor needs either rows near SFT's decision boundary or the margin itself, which the op-margin scorer reads in one forward pass and which moves on saturated rows too.

### The served model lags the trained one

A merged bf16 model drops any part of the LoRA delta below half a bf16 unit of its weight. For a weight around 1e-2 that unit is about 6e-5, and GRPO's delta is small:

| Step | 0 | 20 (smoke) | 50 | 100 | 150 | 187 | r2em-100 |
|---|---|---|---|---|---|---|---|
| Share of the delta kept by the merge | 1.00 | 0.23 | 0.40 | 0.50 | 0.55 | 0.58 | 0.91 |
| Weights left bit-identical | 1.00 | – | 0.87 | 0.83 | 0.81 | 0.79 | 0.53 |

The final adapter's mean |ΔW| is 8.7e-6, against r2em-100's 2.4e-5. The merged and unmerged models still agree on 98.8–99.2% of the held-out decisions, but their margins differ by a median of 2.5–3 nats. So the audit measured SFT plus about 58% of the trained update, which is also the model the rollouts came from. This does not explain the hire-rate drop. The unmerged policy is further along the same direction, and the served monitor read the same hire rates as HF + PEFT throughout.

## Rationales

Thirty attribute-free cases were read: 20 of the 40 CVs that SFT hired and GRPO rejected, and 10 of the 89 both models hired (seed 42; `reports/grpo_pilot/rationale_review.csv`). Each rationale, GRPO's and SFT's, was checked claim by claim against the CV and the job description. A fact counts as wrong only when CV text contradicts it or the claimed item is plainly absent. A year count that depends on the CV's unknown date does not count. A misread is a reason to reject that changes the requirement: a "plus" treated as a must-have, a criterion the job does not state, or a stricter version of one it does.

| | Cases with a wrong fact | Wrong facts | Rejections resting on a misread requirement |
|---|---|---|---|
| GRPO, its 20 new rejections | 5 | 5 | 9 |
| GRPO, the 10 shared hires | 4 | 4 | – |
| SFT, the same 20 (it hired) | 7 | 8 | – |
| SFT, the 10 shared hires | 2 | 2 | – |

GRPO does not invent more facts than SFT (9 against 10 in 30 cases), but its new rejections err in their own way: 9 of 20 rest on a requirement read more strictly than written. In four, a plus becomes a must-have: AngularJS, Rust and Electron where the job says "willing to learn", Firebase "as a plus", printing methods. In two, the criterion is not in the job (EdTech experience, "recent" experience). In three, a real one is raised: "full HR management and payroll", "Java backend", "three years of continuous manual testing".

The wrong facts in its rejections are mostly a skill the CV shows under another name. G06 says the candidate "lacks … offline storage", while the CV lists Room, which GRPO credits as "strong Room" for the same CV in G10. ASP.NET Core counts as no ASP.NET MVC, and a Perl-to-PHP switch as no legacy migration. Only one miscounts tenure: G22 says the candidate lacks "the required three years of continuous manual testing", while the CV shows 01/2019–12/2022 as QA Manual Tester. In 15 of the 20 new rejections the GPT-4o reference hires, and 9 of those 15 GRPO rationales carry a wrong fact or a misread.

SFT's errors run the other way, as overclaims. It credits terms from the job that the CV lacks (Docker, AWS, Agile, performance tuning), and it stretches years to the job's threshold: "5+ years" on a CV that runs from 03.19 to 05.23. In the shared hires, GRPO's rationales are close copies of SFT's and inherit the same habit (Cypress in G18, which only the job mentions).

So GRPO became stricter mostly by reading requirements more strictly, not by inventing facts. The blind review found the DPO models inventing facts in both directions; here, in the reject direction, the change is mostly requirement inflation. The reviewer is one LLM pass (Claude), not blind to the decisions.

## Reading

GRPO did what its reward asked on the prompts it saw. It was symmetric in principle and one-directional in effect. At matched hire rate it is more consistent than SFT by the same 0.95 pp that DPO reached. Its interval is the first among the preference arms to exclude zero. It bought that at round-1 DPO's operating point: 8 pp of hire rate and 2 pp of agreement with the reference. Two optimisers on top of SFT, DPO and now GRPO, meet at about 1 pp on the same anchors. That gap is inside the noise of a single temperature-0.7 reference sample on borderline CVs (`docs/DATA_EVALUATION.md`), which points at the anchor rather than at another optimiser. The rejection-sampling SFT control (`docs/RFT_CONTROL.md`) keeps the anchors and drops the contrast. It gains nothing at matched hire rate (+0.31 pp) and still lowers the hire rate, to 23.7%.

If GRPO is tried again, three changes follow from this run:

- Keep the pull balanced as the policy moves, not only at the start. For example, re-weight prompts by the current group variance, or drop hire rows that have gone to zero variance.
- Monitor the margin on rows near SFT's decision boundary.
- Use a larger update (learning rate or LoRA scale), so that most of it survives the bf16 merge.

## Caveats

- One run, one seed. The matched interval's upper end is −0.05 pp, so a second seed could land on either side of zero.
- The matched comparison sits at 19.8%, where SFT itself has only 2.87% unstable sets. The counterfactual metric is easier to improve at a low hire rate, and the matching is what corrects for that.
- `grpo-02` balances the pull, a change from the half-tipped design first planned (`grpo-01`), made because of the smoke run. The 256 monitor rows and the 119 held-out candidates are the same in both.
- "Right" means the v1 teacher's attribute-free reference, one sample at temperature 0.7. Two independent judges agree on only about 80% of verdicts (`reports/claude_blind_review/`).

## Reproducing

From a Daint login node, one submission at a time, with the worktree as `REPO` and its `src` first on `PYTHONPATH` (the venv's editable install points at the main checkout):

```bash
export REPO=$PWD PYTHONPATH=$PWD/src
python scripts/build_grpo_prompts.py            # CPU, about a minute: artifacts/semisynthetic-v2/grpo-02
CONFIG=configs/mitigation/grpo/qwen3.5-9b_en_only_grpo_smoke_gh200.yaml \
  scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/train_grpo.sbatch
CONFIG=configs/mitigation/grpo/qwen3.5-9b_en_only_grpo_pilot_gh200.yaml \
  scripts/cscs/submit.sh -p normal -t 03:00:00 scripts/cscs/train_grpo.sbatch
CONFIGS=configs/audit/qwen3.5-9b_en_grpo_pilot_gh200.yaml \
  scripts/cscs/submit.sh -p normal -t 01:00:00 scripts/cscs/run_audits.sbatch
VALIDATE=configs/audit/qwen3.5-9b_en_grpo_pilot_gh200.yaml:adapter \
PASSES=configs/audit/qwen3.5-9b_en_grpo_pilot_gh200.yaml:adapter \
COMPARE=configs/audit/qwen3.5-9b_en_grpo_pilot_gh200.yaml:configs/audit/qwen3.5-9b_en_sft_en_only_gh200.yaml \
  scripts/cscs/submit.sh -p normal -t 01:00:00 scripts/cscs/op_control.sbatch
```

These were jobs 5065891 (smoke, 14 minutes), 5066266 (74 minutes), 5066531 (18 minutes) and 5066563 (18 minutes). The run's own records are in `outputs/grpo/qwen3.5-9b_en_only_grpo_pilot/`: `sync_validation.jsonl`, `decision_eval.jsonl`, `decision_served_eval.jsonl` and `decision_eval_sft.json`. Job 5065963 died in vLLM's vision-encoder profiling before training. The engine now runs text-only (`language_model_only`).
