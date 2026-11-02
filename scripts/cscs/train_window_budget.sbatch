#!/bin/bash
# Near-parameter-matched window LoRA; final adapter fixed before training.
# POSTER_SEED=42|43|44. See docs/POSTER_FOLLOWUP_EXPERIMENTS.md.
#SBATCH --job-name=xw-window163
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out
set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
case ${POSTER_SEED:-} in 42|43|44) ;; *) echo "POSTER_SEED must be 42, 43 or 44" >&2; exit 2;; esac
export CONFIG="configs/mitigation/distill/decision_distill_l15_20_r163_seed${POSTER_SEED}.yaml"
cd "$REPO"
"$PY" scripts/verify_window_budget.py --config "$CONFIG" --phase before
# Final adapters are only written after training. Existing checkpoints are resumed
# by train_distill.sbatch; a completed final adapter must never be retrained.
if ! "$PY" - "$XW/outputs/distill/decision_distill_l15_20_r163_seed${POSTER_SEED}" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
done = (p / "run_config.json").exists() and json.loads((p / "run_config.json").read_text()).get("status") == "done"
sys.exit(0 if done and (p / "adapter_model.safetensors").exists() else 1)
PY
then
  bash scripts/cscs/train_distill.sbatch
fi
"$PY" scripts/verify_window_budget.py --config "$CONFIG" --phase after
echo "STAGE window-budget training OK"
