#!/bin/bash
# Operating-point control on one GH200 node: one margin pass per GPU (up to four), then the
# comparisons, which are CPU work on the same node.
#
#   PASSES="cfgA.yaml:adapter cfgB.yaml:adapter cfgA.yaml:base" \
#   COMPARE="cfgB.yaml:cfgA.yaml cfgA.yaml:base" \
#     scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/op_control.sbatch
#
# COMPARE entries are run:against (compare-arms) or cfg:base (the adapter against its base).
#
#SBATCH --job-name=xw-opc
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
read -r -a passes <<< "${PASSES:?set PASSES to up to four cfg:base|adapter entries}"
read -r -a compares <<< "${COMPARE:-}"
[ ${#passes[@]} -le 4 ] || { echo "FATAL: at most four passes, one per GPU"; exit 2; }
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "margin passes: ${passes[*]}"
pids=(); fail=0
for g in "${!passes[@]}"; do
  cfg=${passes[$g]%%:*}; which=${passes[$g]##*:}
  CUDA_VISIBLE_DEVICES=$g $PY scripts/operating_point_control.py margins --config "$cfg" --which "$which" \
    > "$XW/logs/xw-opc-${SLURM_JOB_ID:-local}-gpu$g.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for g in "${!passes[@]}"; do
  echo "--- gpu$g ${passes[$g]}"
  { grep -E "wrote|skipping|Error|Traceback" "$XW/logs/xw-opc-${SLURM_JOB_ID:-local}-gpu$g.log" || true; } | tail -n 3
done
[ "$fail" -eq 0 ] || { echo "FATAL: a margin pass failed -- see the gpu logs"; exit 1; }

for c in "${compares[@]}"; do
  run=${c%%:*}; ref=${c##*:}
  if [ "$ref" = base ]; then
    step "compare $run against its base"
    $PY scripts/operating_point_control.py compare --config "$run"
  else
    step "compare $run against $ref at matched hire rate"
    $PY scripts/operating_point_control.py compare-arms --config "$run" --against "$ref"
  fi
done
echo "STAGE opc OK"
