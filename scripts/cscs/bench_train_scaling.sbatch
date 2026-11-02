#!/bin/bash
# Acceleration benchmark B2 (docs/ACCEL_BENCH.md): decision distillation's first 40 optimizer
# steps, 32 rows a step, on 4 GH200 under DDP and then on 1 GH200 (accumulation 32), one after
# the other on the same node. Each run is one `cost train bench-distill-ddp<n>` step; the
# trainer's train_runtime and train_samples_per_second are in the log. Nothing is kept.
#
#   HBM_COST_ARM=accel_bench REPO=$PWD scripts/cscs/submit.sh scripts/cscs/bench_train_scaling.sbatch
#
#SBATCH --job-name=xw-bench-train
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:45:00
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
for n in 4 1; do
  rm -rf "$XW/outputs/bench/distill_ddp$n"
  step "distillation, 40 steps on $n GPU(s)"
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) cost train "bench-distill-ddp$n" \
    $PY -m torch.distributed.run --standalone --nproc_per_node=$n \
    -m hiring_bias_mitigation.mitigation.decision_distill --config "configs/mitigation/distill/bench_ddp$n.yaml"
done
echo "STAGE bench-train OK"
