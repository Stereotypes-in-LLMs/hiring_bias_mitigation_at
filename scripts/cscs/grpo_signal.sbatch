#!/bin/bash
# Per-class GRPO signal at every checkpoint (scripts/grpo_signal_by_reference.py score): one
# process per GPU, each scoring its share of the checkpoints on the run's training and held-out
# prompts with HuggingFace, served (merged) and trained (PEFT) weights. Already-written outputs
# are skipped, so a run cut by the time limit continues when the same job is submitted again.
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/grpo_signal.sbatch
#
#SBATCH --job-name=xw-grpo-signal
#SBATCH --account=lp193
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv"; exit 3; }
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a) | uncommitted $(git status --porcelain | wc -l)"

step "score checkpoints, one shard per GPU"
pids=(); fail=0
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/grpo_signal_by_reference.py score --shard $g --num-shards 4 \
    > "$XW/logs/xw-grpo-signal-${SLURM_JOB_ID:-local}-gpu$g.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for g in 0 1 2 3; do
  echo "--- gpu$g"
  { grep -E "shard|wrote|Error|Traceback" "$XW/logs/xw-grpo-signal-${SLURM_JOB_ID:-local}-gpu$g.log" || true; } | tail -n 6
done
[ "$fail" -eq 0 ] || { echo "FATAL: a shard failed -- see the gpu logs"; exit 1; }
echo "STAGE grpo-signal OK"
