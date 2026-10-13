#!/bin/bash
# The v2 small batch on one GH200 node, in two stages that each fit the debug partition's
# 30-minute cap:
#
#   generate  the teacher's negatives and judgements on all four GPUs (tensor parallel), then the
#             GPU-free direction checks (D1-D3, D5-D7). Every teacher pass is dumped under raw/
#             and resumed from there, so a stage cut short by the time limit loses nothing
#             that finished -- resubmit and it continues.
#   check     the direction check with the SFT student and the base model (D4, HF + PEFT).
#
# Needs scripts/cscs/download.sbatch and scripts/cscs/build_env.sbatch to have finished. Runs
# offline: every model is already in $HF_HOME. One node, one task, one submission.
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/small_batch.sbatch generate
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/small_batch.sbatch check
#   scripts/cscs/submit.sh scripts/cscs/small_batch.sbatch all   # normal, both stages
#
#SBATCH --job-name=xw-v2-small
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
STAGE=${1:-all}
BATCH=artifacts/semisynthetic-v2/batch-01
CONFIG=configs/generation/negatives_v2_small_gh200.yaml
STUDENT=Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | stage $STAGE | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

if [[ $STAGE == generate || $STAGE == all ]]; then
  step "generate 1/2 teacher: negatives + judgements (resumes from raw/ dumps)"
  $PY scripts/generate_negatives_v2.py --config "$CONFIG"
  step "generate 2/2 direction check, GPU-free part (D1-D3, D5-D7)"
  $PY scripts/check_preference_direction.py --batch "$BATCH"
fi

if [[ $STAGE == check || $STAGE == all ]]; then
  step "check: SFT student and base model preferences (D4, HF + PEFT, one GPU)"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/check_preference_direction.py --batch "$BATCH" \
    --student "$STUDENT" --also-base --batch-size 4
fi

echo "STAGE $STAGE OK -> $XW/$BATCH (read samples.md and direction_report.md)"
