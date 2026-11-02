#!/bin/bash
# Acceleration benchmark B2, repeat of the 1-GPU run (docs/ACCEL_BENCH.md, "B2 repeat"): the
# first run slowed from 50 to 62 s a step after step 5. The same 40 steps on one GH200, with
# the GPU's clocks, power, temperature and throttle reasons logged every 10 s, and
# `nvidia-smi -q -d PERFORMANCE,CLOCK` before and after. The node is allocated whole.
#
#   HBM_COST_ARM=accel_bench REPO=$PWD scripts/cscs/submit.sh scripts/cscs/bench_train_1gpu_repeat.sbatch
#
#SBATCH --job-name=xw-bench-1gpu
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:15:00
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
MON="$XW/logs/xw-bench-1gpu-${SLURM_JOB_ID:-none}"
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit --format=csv
nvidia-smi -q -i 0 -d PERFORMANCE,CLOCK,POWER > "$MON-before.txt"
# Every 10 s: the basic counters as CSV, and the full PERFORMANCE section (throttle reasons).
nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used \
  --format=csv -l 10 > "$MON-gpu.csv" &
SAMPLER=$!
( while true; do date +%FT%T; nvidia-smi -q -i 0 -d PERFORMANCE; sleep 10; done ) > "$MON-perf.txt" &
PERF=$!
trap 'kill $SAMPLER $PERF 2>/dev/null || true' EXIT
rm -rf "$XW/outputs/bench/distill_ddp1"
step "distillation, 40 steps on 1 GPU (repeat)"
CUDA_VISIBLE_DEVICES=0 cost train "bench-distill-ddp1-r2" \
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 \
  -m hiring_bias_mitigation.mitigation.decision_distill --config configs/mitigation/distill/bench_ddp1.yaml
nvidia-smi -q -i 0 -d PERFORMANCE,CLOCK,POWER > "$MON-after.txt"
echo "STAGE bench-1gpu OK"
