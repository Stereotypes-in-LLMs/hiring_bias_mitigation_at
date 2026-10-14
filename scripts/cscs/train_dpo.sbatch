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
mkdir -p "$WANDB_DIR"
CONFIG=${CONFIG:-configs/mitigation/dpo/qwen3.5-9b_en_only_dpo_v2neg_r1_gh200.yaml}
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | config $CONFIG | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

# Offline runs from earlier segments go up first, oldest first, so the resumed online run
# continues one curve. No route out of the node: skip, the trainer then logs offline too.
if [ -n "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-online}" != offline ]; then
  step "W&B: upload offline runs not synced yet"
  shopt -s nullglob; offline=("$WANDB_DIR"/wandb/offline-run-*); shopt -u nullglob
  if ! curl -s -m 10 -o /dev/null https://api.wandb.ai; then
    echo "no route to api.wandb.ai from $(hostname): this segment logs offline"
  elif [ ${#offline[@]} -gt 0 ]; then
    timeout 300 "$XW/venv/bin/wandb" sync --yes "${offline[@]}" \
      || echo "wandb sync failed; the runs stay offline and the next segment tries again"
  fi
fi

step "DPO on 4 GPUs (DDP); resumes from the newest finished checkpoint if there is one"
$PY -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m hiring_bias_mitigation.mitigation.dpo --config "$CONFIG"
echo "STAGE dpo OK"
