#!/bin/bash
# DPO on one GH200 node, four GPUs under torchrun (DDP). With `resume: true` in the config the
# run continues from its newest finished checkpoint, so a schedule longer than debug's 30
# minutes is the same job submitted again, one submission at a time:
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/train_dpo.sbatch
#   CONFIG=configs/mitigation/dpo/<x>.yaml scripts/cscs/submit.sh -p debug -t 00:30:00 \
#     scripts/cscs/train_dpo.sbatch
#
#SBATCH --job-name=xw-dpo
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline WANDB_DIR=$XW/wandb
mkdir -p "$WANDB_DIR"
CONFIG=${CONFIG:-configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1_gh200.yaml}
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | config $CONFIG | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "DPO on 4 GPUs (DDP); resumes from the newest finished checkpoint if there is one"
$PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m hiring_bias_mitigation.mitigation.dpo --config "$CONFIG"
echo "STAGE dpo OK"
