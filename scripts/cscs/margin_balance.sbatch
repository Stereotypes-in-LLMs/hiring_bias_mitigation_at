#!/bin/bash
# Round-2 DPO data on one GH200 node: prepare, the SFT student's margins on four GPUs, select,
# then the GPU-free direction checks on the result (scripts/margin_balance.py).
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/margin_balance.sbatch
#
#SBATCH --job-name=xw-margin
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
OUT=artifacts/semisynthetic-v2/onpolicy-hire32/margin_balanced
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "prepare"
cost datagen prepare $PY scripts/margin_balance.py prepare
step "score: SFT margins, one process per GPU"
pids=(); fail=0
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g cost datagen score \
    $PY scripts/margin_balance.py score --shard $g --shards 4 \
    > "$XW/logs/xw-margin-${SLURM_JOB_ID:-local}-gpu$g.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for g in 0 1 2 3; do echo "--- gpu$g"; { grep -E "scored|skipping|Error|Traceback" "$XW/logs/xw-margin-${SLURM_JOB_ID:-local}-gpu$g.log" || true; } | tail -n 3; done
[ "$fail" -eq 0 ] || { echo "FATAL: a scoring process failed -- see the gpu logs"; exit 1; }
step "select"
cost datagen select $PY scripts/margin_balance.py select
step "direction check, GPU-free part"
cost datagen direction-check \
  $PY scripts/check_preference_direction.py --pairs "$OUT/dpo_v2_candidates.parquet"
echo "STAGE margin OK"
