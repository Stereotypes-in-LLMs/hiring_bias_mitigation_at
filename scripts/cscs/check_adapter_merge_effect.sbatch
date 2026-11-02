#!/bin/bash
# Three independent arms on three GH200s. Inference only; no checkpoint writes.
#SBATCH --job-name=xw-merge-effect
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out
set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
cd "$REPO"
export HBM_COST_ARM=merge_precision
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODE=${MERGE_MODE:-pilot}
SEED=${MERGE_SEED:-42}
case "$MODE" in pilot|full) ;; *) exit 2;; esac
case "$SEED" in 42|43|44) ;; *) exit 2;; esac
if [[ "$MODE" == pilot && "$SEED" != 42 ]]; then exit 2; fi
BATCH=${MERGE_BATCH:-8}
OUT_ROOT=${MERGE_OUT_ROOT:-$REPO/reports/research_followup/merge_precision}
ARMS=(full_r32 window_r32 window_r163)
PIDS=()
for GPU in 0 1 2; do
  ARM=${ARMS[$GPU]}
  (
    export CUDA_VISIBLE_DEVICES=$GPU
    cost eval "merge-precision-${MODE}-${ARM}-s${SEED}" \
      "$PY" scripts/check_adapter_merge_effect.py --arm "$ARM" --seed "$SEED" \
      --mode "$MODE" --batch-size "$BATCH" --device cuda:0 --out-root "$OUT_ROOT"
  ) > "$XW/logs/merge-precision-${SLURM_JOB_ID}-${ARM}.out" 2>&1 &
  PIDS+=("$!")
done
DIAG_PID=
if [[ "$MODE" == full && "$SEED" == 42 && ${MERGE_BATCH_DIAG:-0} == 1 ]]; then
  (
    export CUDA_VISIBLE_DEVICES=3
    export HBM_COST_ARM=mechanism_batch_diagnostic
    export PYTHONPATH="$REPO/src:$XW/pylib-fla${PYTHONPATH:+:$PYTHONPATH}"
    export FLA_DISABLE_BACKEND_DISPATCH=1
    cost patching mechanism-batch-diagnostic \
      timeout 900 "$PY" scripts/diagnose_mechanism_batch.py \
      --spec dpo4880_s43 --contrast mil_combat_civ_expl --pairs 8 --repeats 2 \
      --device cuda:0 --out reports/research_followup/mechanism_seeds/batch_diagnostic
  ) > "$XW/logs/merge-precision-${SLURM_JOB_ID}-batch-diagnostic.out" 2>&1 &
  DIAG_PID=$!
fi
STATUS=0
for PID in "${PIDS[@]}"; do wait "$PID" || STATUS=1; done
# This waits only on the three merge workers. A diagnostic failure cannot prevent
# successful full scores from reaching the CPU aggregate.
if [[ "$MODE" == full && "$STATUS" == 0 ]]; then
  "$PY" scripts/summarize_adapter_merge_effect.py --source "$OUT_ROOT" \
    --out "$OUT_ROOT/aggregate" --input-root "$XW" --when-ready || STATUS=1
fi
if [[ -n "$DIAG_PID" ]]; then
  DIAG_STATUS=0
  wait "$DIAG_PID" || DIAG_STATUS=$?
  "$PY" - "$OUT_ROOT" "$SLURM_JOB_ID" "$DIAG_STATUS" <<'PY'
import json, sys
from pathlib import Path
root, job, status = sys.argv[1:]
Path(root, f"diagnostic-job-{job}.json").write_text(json.dumps({
    "job_id": job, "diagnostic_exit_code": int(status),
    "merge_aggregation_independent": True,
    "log": f"merge-precision-{job}-batch-diagnostic.out",
}, indent=2) + "\n")
PY
fi
exit "$STATUS"
