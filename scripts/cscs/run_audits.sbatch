#!/bin/bash
# Benchmark audits on one GH200 node, one config per GPU (up to four), each through vLLM with
# adapters served as merged weights (merge.py merges a DPO checkpoint's SFT parent first).
# Generations are cached in chunks under outputs/raw, so a run cut by the time limit continues
# where it stopped when the same job is submitted again.
#
#   CONFIGS="configs/audit/a.yaml configs/audit/b.yaml" \
#     scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/run_audits.sbatch
#
#SBATCH --job-name=xw-audit
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
read -r -a configs <<< "${CONFIGS:?set CONFIGS to up to four audit configs}"
[ ${#configs[@]} -le 4 ] || { echo "FATAL: at most four configs, one per GPU"; exit 2; }
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "audits: ${configs[*]}"
pids=(); fail=0
for g in "${!configs[@]}"; do
  name=$(basename "${configs[$g]}" .yaml)
  CUDA_VISIBLE_DEVICES=$g $PY scripts/run_audit.py --config "${configs[$g]}" --no-similarity \
    > "$XW/logs/xw-audit-${SLURM_JOB_ID:-local}-$name.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for c in "${configs[@]}"; do
  name=$(basename "$c" .yaml); echo "--- $name"
  { grep -E "Traceback|Error|written|unstable|parse" "$XW/logs/xw-audit-${SLURM_JOB_ID:-local}-$name.log" || true; } | tail -n 5
done
[ "$fail" -eq 0 ] || { echo "FATAL: an audit failed or was cut -- see the per-config logs"; exit 1; }
echo "STAGE audit OK"
