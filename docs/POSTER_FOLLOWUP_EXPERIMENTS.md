# Poster follow-up experiments

Registered on 2026-11-02 after the poster audit, before these new runs. The user
requested a window LoRA with the full adapter's parameter budget and mechanism
checks for the remaining seeds. The development benchmark remains exploratory.

## Parameter-budget control

The original layers 15-20 adapter has 11,403,264 trainable parameters at rank 32.
The full-layer rank-32 adapter has 58,195,968. The nearest uniform integer rank
for the same window and seven target module types is **163**:

    11,403,264 / 32 * 163 = 58,085,376

This is 110,592 parameters, or **0.1900%**, below the full adapter. Rank 164 would
be 58,441,728, or 0.4223% above. This is a near-matched budget, not an exact match.
The registered tolerance is 0.2%. Alpha is 326, so alpha/r remains 2. No trainer
change or per-module rank pattern is introduced solely to eliminate that gap.

Seeds 42, 43 and 44 each use one epoch, the original 341-step schedule, the same
data, learning rate, optimizer and batch settings as the same-seed full adapter.
Only rank and layer placement differ from the full arm. The final adapter is
the declared checkpoint, not the best checkpoint found using audit results.

Files:

- `configs/mitigation/distill/decision_distill_l15_20_r163_seed{42,43,44}.yaml`
- matching `configs/audit/qwen3.5-9b_en_decision_distill_l15_20_r163_seed{42,43,44}_gh200.yaml`
- `scripts/verify_window_budget.py`
- `scripts/cscs/train_window_budget.sbatch`
- `scripts/cscs/evaluate_window_budget.sbatch`

The CPU preflight reads the existing same-seed adapters' tensor shapes and checks
their actual budgets and module counts. After training it verifies the final
adapter has exactly 58,085,376 parameters on layers 15-20 and a completed run
manifest. The audit merges adapters using the established pipeline and records
merge retention before scoring. Comparisons against SFT-v3 and the same-seed
full-layer adapter are both saved. Review training loss, merge retention and
reference-agreement guardrails alongside matched instability.

Primary comparison: window-r163 minus same-seed full-layer-r32 in instability at
the full arm's own greedy hire count, with the window arm matched to that count
and the existing candidate-bootstrap CI. The canonical `compare-arms` invocation
uses full as `--config` and window as `--against`. Its full-minus-window estimate
is negated, and its CI endpoints reversed and negated, by
`scripts/window_budget_results.py`. That script also reports full-r32, window-r32
and window-r163 at the common SFT-v3 hire count, approximately 22.512%, after
scoring. It reads the exact SFT count rather than rounding a displayed percentage.
Each seed writes its own directory under `reports/experiment_followup/window_seedN`.
Positive means the window arm is worse. This is a two-sided comparison: report
the estimate and interval regardless of direction. A CI crossing zero is not a
proof of equivalence. Do not select another rank, epoch or checkpoint in response
to the benchmark and then relabel the result as confirmatory.

A worse window result would reduce the simple parameter-count explanation. It
would not establish that localization generally cannot guide editing: layerwise
rank, optimization geometry and the higher rank's merge behavior still differ.
Matching performance would show that capacity contributed to the earlier result.

### Cost and bounds

The previous r32 window runs used 4.80-4.88 GPU-h **per seed**, not for all three.
Plan approximately 15 used GPU-h for these three training runs before evaluation,
with uncertainty from the larger rank. This is an estimate, not a measured result.
Each training job is capped at 2 hours on four GH200s; each dependent evaluation
at 1 hour on four GH200s. The three training and three evaluation caps total
36 allocated GPU-h if every job reaches its cap. No automatic retry is scheduled.

## Additional mechanism seeds

The frozen existing checkpoints are:

| Spec | Trained checkpoint |
|---|---|
| `dpo4880_s43` | `outputs/dpo/qwen3.5-9b_en_only_dpo_v3_4880_seed43/checkpoint-305` |
| `dpo4880_s44` | `outputs/dpo/qwen3.5-9b_en_only_dpo_v3_4880_seed44/checkpoint-305` |
| `grpo_r4_s43` | `outputs/grpo/grpo_improve_r4_tipped_seed43` |
| `grpo_r4_s44` | `outputs/grpo/grpo_improve_r4_tipped_seed44` |
| `distill42` | `outputs/distill/decision_distill_seed42` |
| `distill44` | `outputs/distill/decision_distill_seed44` |

Each uses the same SFT-v3 adapter as the reference. New specs are added to
`configs/interp/models.yaml`; `02_margins.py` maps them to their own behavioural
audit so a full-grid margin pass must reproduce 31,050 rows with at least 97%
decision agreement. A completed pass below that threshold stops the job.

The four contrasts are combat/civilian implicit, combat/civilian explicit,
atheist/Christian implicit and male/female implicit. Every checkpoint uses pairs
selected on SFT-v3's effect, requested n=64 and selection seed 42. Gender has 45
eligible pairs. The exact `pair_uid` set must equal the corresponding SFT-v3
reference list; no checkpoint-specific selection is used. The default patching
pipeline includes real component interventions, AtP and real head verification
with random controls. AtP ranks and verified intervention ranks remain distinct.

`poster_mechanism.sbatch` reuses valid completed full-grid margins and each
completed contrast. This includes the already run distill42 contrasts. Partial
artifacts resume through the established scripts. New caches record adapter
fingerprints before their first compute; changed or ambiguous partial caches
stop the job. Existing distill42 patch caches are explicitly grandfathered with
`legacy_cache_adapter_identity_unverified` provenance, not claimed to have a
verified historical adapter fingerprint. Completed margins also check their
native adapter fingerprint. It first runs any missing
margin pass on one GPU, then missing contrasts in parallel on up to four GPUs,
and finally produces the existing `20_distill_compare.py` summaries.

The rough additional used-GPU estimate is 10-12 hours before reusing old work.
The one-node-per-checkpoint packaging uses more allocated than used GPU-h during
its single-GPU margin pass. Each of six jobs is capped at 1.5 hours on four GPUs,
36 allocated GPU-h in the worst case. No additional training is part of these
six jobs, and no automatic retry is scheduled.

## Submitting and inspecting

From the remote repository, first inspect the read-only plan:

```bash
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw \
  ../venv/bin/python scripts/submit_poster_followups.py
```

After source checks and CPU preflight are reviewed, the user-authorized submits:

```bash
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw \
  ../venv/bin/python scripts/submit_poster_followups.py --group windows --submit
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw \
  ../venv/bin/python scripts/submit_poster_followups.py --group mechanisms --submit
```

The launcher checks the live queue and records accepted job IDs immediately in
`reports/experiment_followup/submissions.json`. It reuses live jobs by unique
name and completed final artifacts. It refuses to resubmit a recorded job whose
result has not been reviewed, including failures. Training-to-evaluation uses
Slurm `afterok`; a failed training run cannot silently proceed to scoring.

The existing DDP single-GPU repeat is separate and must not be resubmitted by
this launcher. Its reviewed result needs to be incorporated into the poster
generation and cost ratio separately.
