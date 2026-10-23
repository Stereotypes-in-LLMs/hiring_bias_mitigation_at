# GRPO-v3-all checkpoint benchmark trajectory

## Setup

This inference-only diagnostic was requested on 2026-10-23. It asks when GRPO-v3-all's
benchmark TPR fell relative to its SFT-v3 parent, whether an 8 pp training alarm would have
acted before the existing 10 pp evaluation guardrail, and how checkpoint 525 should be read.
No training is authorized by this experiment. The existing EVAL_STANDARD is unchanged.

The hypothesis is that TPR already fell at an earlier saved checkpoint while the original
projected monitor hid that loss. Evidence against it would be stable direct TPR until the
final saved steps. This is a descriptive trajectory, not a new confirmatory comparison.
All 450 benchmark pairs have already been evaluated during development. Selecting a step
from this trajectory is exploratory and cannot restore an untouched final test.

Control: SFT-v3 final adapter, run
`Qwen3.5-9B--en--sft--adapter--en_only_sft_v3_additional`.
Parent adapter SHA256 is
`d8d63526a43dab37a1fb41a7d7a3541f6a023c4fa2a19e9409e04da331e419ee`.
GRPO run: `outputs/grpo/qwen3.5-9b_en_only_grpo_v3_all`, 9,140 training rows.
Base revision: `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.

The 20 retained checkpoints are 100, 125, 150, 175, 200, 225, 250, 275, 300, 325,
350, 375, 400, 425, 450, 475, 500, 525, 550, and 571. Steps 25, 50, and 75 were
not retained. A crossing before step 100 can only be bracketed between step 0 and 100.
The final root adapter equals checkpoint 571 byte for byte, SHA256
`9840fb1e6ab7f1b4df6689082504d5a8f254bb952f25e2fd85dfc0eb5a817f0a`.
Parent, checkpoint 525, and final 571 have complete existing audits and are imported after
checking row keys, labels, full prompt reconstruction, decoding, model chain and software.
The other 18 checkpoints require inference. No unlisted checkpoints are added automatically.

Primary outcome: the existing generated-decision `tpr_vs_gpt4o_pct`, on 31,050 rows including
450 attribute-free controls, subtracting the parent's value on the same scope. The repository
metric conditions on usable decisions. Full-reference-denominator TPR and parse coverage are
reported alongside it, so missing output cannot silently improve a curve. FPR and agreement
are descriptive. Existing parent/525/final op-margin-v2 outputs remain a secondary cross-check;
new margin passes are omitted because they almost double this round's inference time.

Protocol: original binary baseline prompt, greedy generation, thinking disabled, 512-token
budget, seed 42, original submission order and 4,000-row chunks, merged bf16 models,
vLLM 0.31.0, Torch 2.13.0, Transformers 5.17.0. Tensor parallel size is 1, matching the
original audit. Every GRPO merge applies SFT first and requires at least 80% retention of
the GRPO adapter's own update. Full base/input/adapter hashes are bound to the manifest.

The trajectory displays both policies without redefining either: an 8 pp training alarm
at `delta_tpr_pp <= -8`, and the existing final guardrail failure at `delta_tpr_pp < -10`.
Exactly -10 pp remains inside the existing "at most 10 pp" guardrail. The trajectory alone
does not calibrate a new benchmark-dev monitoring rule or make checkpoint 525 confirmatory.

The direct benchmark-dev partition is frozen separately before new checkpoint inference.
`outputs/benchmark_dev_monitor_v1/split.json` SHA256 is
`b5fe8fc48ee05ab4e0ef0d2f0101c9bdbddce7e5b2f07ddcbffc993e61035d31`;
its manifest SHA256 is
`e8d073849e56ed9b35ccf8f29d370e1e333d62c47ef947187addf281e489d2f7`.
The candidate-disjoint partitions contain 90 dev candidates and 60 historical-remainder
candidates, respectively 18,630 and 12,420 prompts. Neither partition is renamed a fresh test.
The trajectory locks these files, checks full candidate/pair membership, and reports full,
dev, and historical-remainder TPR, relative TPR, paired positive-label loss/gain, and invalid
counts separately at each checkpoint. There is no projection or reweighting between them.

Three rule columns remain distinct: `full_8pp_alert`, `full_fails_10pp_guardrail`, and
`dev_8pp_candidate_alert`. Cached checkpoint 525 changes full TPR by -9.29469 pp but dev TPR
by -6.74770 pp; final 571 changes them by -10.10628 and -7.90089 pp. Thus neither cached dev
endpoint reaches -8 pp. The unfinished intermediate trajectory may contain an earlier dev
crossing, so endpoints alone do not establish that an 8 pp dev rule would never stop.
For each rule the summary records the first observed crossing and the immediately preceding
retained safe checkpoint. Until all earlier retained checkpoints are scored, that crossing
is explicitly provisional. A later recovery never revokes a first historical stop.

Stops: any hash/version/key/label mismatch, merge retention below 80%, inference exception,
or over 1% unusable outputs. A failure is not skipped in favour of later checkpoints.
Slurm wall time is 4 hours and one submitted job owns the complete remaining trajectory.
Four independent TP1 workers each process their assigned checkpoints sequentially, with
8 CPU threads per merge and at most one model per GPU. The original model parallelism and
decoding remain unchanged. Expected wall time is approximately 1.5–2 hours, based on the
existing 13.5-minute generation pass plus merge and load time. Temporary merge disk usage
peaks near 76 GB. Each worker writes a separate progress file and inference log; only the
parent process updates the combined trajectory and progress files.
Completed chunks and per-checkpoint results survive a timeout. Resume uses the same frozen
manifest. Derived temporary merges are deleted only after complete raw scores are saved
atomically and hashed; original adapters, merges, raw audits, and published tables are untouched.

## Result

Pending inference. Read `trajectory.json`, `progress.json`, and each
`steps/step-NNNN/result.json` under
`$HBM_OUTPUT_ROOT/outputs/checkpoint_benchmark/grpo_v3_all_20261023/`.
Every result records its raw output hash, adapter chain, merge retention, software, commit,
Slurm job ID, and elapsed time. Scores become visible after each checkpoint completes.

## Reading

Compare direct generated TPR with SFT-v3 and report the first observed 8 pp and 10 pp crossing,
including any recovery or subsequent crossing. A saved-step trajectory cannot identify the
exact optimizer step between checkpoints. Keep the checkpoint-525 fairness claim separate
from the claim that a monitor safely selected it. Its benchmark result is already exploratory.

## Caveats

The 31,050 rows represent 150 candidates and 450 pairs, not 31,050 independent examples.
Future benchmark-dev monitoring needs a frozen candidate split and paired candidate inference;
it cannot be called an untouched test after this trajectory. Greedy generation and forced
decision margins are different readouts and must not be combined in a single TPR curve.

## Reproducing

After committing the runner, config, and this preregistration:

```bash
HBM_OUTPUT_ROOT=/ritom/scratch/cscs/zhang/xw uv run --no-project \
  --python /ritom/scratch/cscs/zhang/xw/venv/bin/python python \
  scripts/checkpoint_benchmark_trajectory.py prepare \
  --config configs/analysis/grpo_v3_all_checkpoint_trajectory.json
REPO=$PWD scripts/cscs/submit.sh scripts/cscs/checkpoint_benchmark_trajectory.sbatch
```

Submit one job, then wait for it to finish. This script never submits further jobs.
The script's `summarize --output-dir <directory>` command is CPU-only and does not modify
the published benchmark reports. No training command is present in this workflow.
