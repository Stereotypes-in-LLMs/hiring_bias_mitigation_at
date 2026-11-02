#!/bin/bash
# Evaluate the declared final window-r163 adapter against SFT-v3 and the
# same-seed all-layer adapter. Submit only after successful training.
#SBATCH --job-name=xw-window163-eval
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out
set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
case ${POSTER_SEED:-} in 42|43|44) ;; *) echo "POSTER_SEED must be 42, 43 or 44" >&2; exit 2;; esac
cd "$REPO"
"$PY" scripts/verify_window_budget.py --phase after \
  --config "configs/mitigation/distill/decision_distill_l15_20_r163_seed${POSTER_SEED}.yaml"
export AUDIT="configs/audit/qwen3.5-9b_en_decision_distill_l15_20_r163_seed${POSTER_SEED}_gh200.yaml"
export AGAINST=configs/audit/qwen3.5-9b_en_sft_v3_additional_gh200.yaml
bash scripts/cscs/evaluate_arm.sbatch
# compare-arms keeps --config at its own rate and moves --against to it.
# Thus this canonical report is full-minus-window at full's rate; the CPU
# follow-up below reverses estimate and CI endpoints to window-minus-full.
cost eval "full-vs-window-r163-s${POSTER_SEED}" \
  "$PY" scripts/operating_point_control.py compare-arms \
  --config "configs/audit/qwen3.5-9b_en_decision_distill_seed${POSTER_SEED}_gh200.yaml" \
  --against "$AUDIT"
"$PY" scripts/window_budget_results.py --seed "$POSTER_SEED"
mkdir -p reports/experiment_followup
"$PY" - "$POSTER_SEED" "$AUDIT" <<'PY'
import json, os, sys
from pathlib import Path
from hiring_bias_mitigation.interp.spec import adapter_fingerprint
seed, config = sys.argv[1:]
adapter = Path(os.environ["HBM_OUTPUT_ROOT"]) / f"outputs/distill/decision_distill_l15_20_r163_seed{seed}"
Path(f"reports/experiment_followup/window_evaluation_seed{seed}.json").write_text(
    json.dumps({"seed": int(seed), "audit_config": config,
                "adapter_fingerprint": adapter_fingerprint(str(adapter)),
                "completed": True, "comparisons": ["sft_v3", "same_seed_full_layer"]}, indent=2)+"\n")
PY
echo "STAGE window-budget evaluation OK"
