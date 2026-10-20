#!/bin/bash
# Shared-seed check of v1's teacher anchors (scripts/anchor_seed_check.py): the 122B teacher on
# all four GPUs (tensor parallel 4), on vLLM's classic model runner, draws v1's 3,000 English
# reference prompts twice -- once under one shared seed 42 (v1's own generate() call), once with
# per-prompt seeds. The teacher takes about 5 minutes to load; each arm should take a few.
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/anchor_seed_check.sbatch
#
# Writes $XW/artifacts/semisynthetic-v3/anchors/seedcheck_{shared42,per_prompt}.parquet and a
# .meta.json beside each. A job that times out after the first arm resumes with the second.
#
#SBATCH --job-name=xw-anchor-seed
#SBATCH --account=lp193
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
export REPO=/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at-seedcheck
source "$REPO/scripts/cscs/env.sh"
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
# The point of the check: the classic runner's per-request torch.Generator. vLLM 0.31 picks
# the V2 runner unless told otherwise (config/vllm.py use_v2_model_runner).
export VLLM_USE_V2_MODEL_RUNNER=0
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD) | dirty $(git status --porcelain | wc -l) | VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER"

step "two arms on the classic runner: shared seed 42, then per-prompt seeds"
$PY scripts/anchor_seed_check.py

step "runner check: the log must not say 'Using V2 Model Runner'"
LOG=$XW/logs/${SLURM_JOB_NAME:-xw-anchor-seed}-${SLURM_JOB_ID:-none}.out
if grep -q "Using V2 Model Runner" "$LOG"; then
  echo "FATAL: the V2 runner ran; both arms are invalid for this check"; exit 4
fi
echo "no 'Using V2 Model Runner' line in $LOG"
echo "STAGE anchor-seed-check OK"
