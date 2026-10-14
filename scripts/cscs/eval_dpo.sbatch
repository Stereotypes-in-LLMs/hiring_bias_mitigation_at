#!/bin/bash
# Decision-level evaluation of every checkpoint of a DPO run on one GH200 node, then W&B:
#
#   RUN=outputs/dpo/<name> scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/eval_dpo.sbatch
#
# One process per GPU over its share of the checkpoints (step 0 = the SFT start); finished
# steps are skipped, so a cut job resumes. Then the curve goes to W&B, and the direction
# checks of DATA_DIRS go up as data-check runs.
#
#SBATCH --job-name=xw-dpo-eval
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
RUN=${RUN:-outputs/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1}
DATA_DIRS=${DATA_DIRS:-artifacts/semisynthetic-v2/onpolicy-hire32/balanced}
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | run $RUN | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "W&B: upload offline runs not synced yet"
wandb_upload_offline

step "decision eval: one process per GPU"
pids=(); fail=0
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_dpo_checkpoints.py --run "$RUN" --shard $g --shards 4 \
    > "$XW/logs/xw-dpo-eval-${SLURM_JOB_ID:-local}-gpu$g.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for g in 0 1 2 3; do echo "--- gpu$g"; { grep -E "step [0-9]+:|nothing left|Error|Traceback" "$XW/logs/xw-dpo-eval-${SLURM_JOB_ID:-local}-gpu$g.log" || true; } | tail -n 4; done
[ "$fail" -eq 0 ] || { echo "FATAL: an eval process failed -- see the gpu logs"; exit 1; }

step "decision eval: summary and W&B"
$PY scripts/eval_dpo_checkpoints.py --run "$RUN" --log

step "data checks to W&B: $DATA_DIRS"
for d in $DATA_DIRS; do $PY scripts/log_data_report.py --dir "$d"; done
echo "STAGE eval OK"
