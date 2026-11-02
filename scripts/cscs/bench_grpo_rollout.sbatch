#!/bin/bash
# Acceleration benchmark B3 (docs/ACCEL_BENCH.md): six GRPO steps (R4's recipe, 256 completions
# a step, 4 GH200 under DDP) with the rollouts from the colocated vLLM engine, then the same six
# steps with the rollouts from transformers' generate. Each run is one
# `cost train bench-grpo-rollout-<path>` step; train_runtime is in the log. Nothing is kept.
#
#   HBM_COST_ARM=accel_bench REPO=$PWD scripts/cscs/submit.sh scripts/cscs/bench_grpo_rollout.sbatch
#
#SBATCH --job-name=xw-bench-grpo
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$WANDB_DIR"
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | repo $REPO | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | uncommitted $(git status --porcelain -- src scripts configs 2>/dev/null | wc -l) files"
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit --format=csv
for path in vllm hf; do
  rm -rf "$XW/outputs/bench/grpo_rollout_$path"
  step "GRPO, 6 steps, rollouts through $path"
  cost train "bench-grpo-rollout-$path" \
    $PY -m torch.distributed.run --standalone --nproc_per_node=4 \
    -m hiring_bias_mitigation.mitigation.grpo --config "configs/mitigation/grpo_improve/bench_rollout_$path.yaml"
done
echo "STAGE bench-grpo OK"
