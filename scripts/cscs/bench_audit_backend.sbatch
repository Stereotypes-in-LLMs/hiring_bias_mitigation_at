#!/bin/bash
# Acceleration benchmark B1 (docs/ACCEL_BENCH.md): the audit's generation on one GH200 through
# the audits' engine (vLLM 0.31, prefix cache on, as-built order) against the repository's HF
# transformers generate loop (batches of 8 and of 32), 45 pairs = 3,105 prompts of the base
# model, two repeats of each condition, each on a GPU of its own (two waves of three). A
# warm-up run on each GPU first fills the node's file and compile caches. Each timed run is one
# `cost eval throughput-accel-<condition>` step. Nothing is scored.
#
#   HBM_COST_ARM=accel_bench REPO=$PWD scripts/cscs/submit.sh scripts/cscs/bench_audit_backend.sbatch
#
# Then, on the login node: scripts/cost_report.py; scripts/throughput_b.py speedup --round accel;
# scripts/throughput_b.py compare --round accel.
#
#SBATCH --job-name=xw-bench-audit
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source "${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}/scripts/cscs/env.sh"
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
"$PY" -c "import hiring_bias_mitigation.utils.cost" \
  || { echo "FATAL: the cost recorder does not import from $REPO/src"; exit 3; }
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | repo $REPO | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | uncommitted $(git status --porcelain -- src scripts configs 2>/dev/null | wc -l) files"
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit,temperature.gpu --format=csv

conditions=(hf8 hf32 vllm)
one() {  # $1 gpu, $2 condition, $3 repeat
  local g=$1 c=$2 r=$3 cfg="configs/throughput/accel/qwen3.5-9b_en_accel-$2.yaml"
  local log="$XW/logs/xw-bench-audit-${SLURM_JOB_ID:-local}-$c-$r.log"
  echo "$r $c on GPU $g -> $log"
  CUDA_VISIBLE_DEVICES=$g "$PY" scripts/throughput_b.py run --warmup --repeat "$r" --config "$cfg" > "$log.warmup" 2>&1
  CUDA_VISIBLE_DEVICES=$g cost eval "throughput-accel-$c" \
    "$PY" scripts/throughput_b.py run --repeat "$r" --config "$cfg" > "$log" 2>&1
}
for r in r1 r2; do
  step "repeat $r"
  pids=(); fail=0
  for i in "${!conditions[@]}"; do
    c=${conditions[$i]}
    if [ "$r" = r1 ]; then g=$i; else g=$(( (i + 2) % 4 )); fi
    one "$g" "$c" "$r" & pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  for c in "${conditions[@]}"; do
    echo "--- $c $r"
    { grep -E "Traceback|Error|prompts, wall|cost: eval" "$XW/logs/xw-bench-audit-${SLURM_JOB_ID:-local}-$c-$r.log" || true; } | tail -n 4
  done
  [ "$fail" -eq 0 ] || { echo "FATAL: a timing run failed in $r -- see its log"; exit 1; }
done
step "greedy decisions across conditions"
"$PY" scripts/throughput_b.py compare --round accel
echo "STAGE bench-audit OK"
