#!/bin/bash
# Operating-point control on one GH200 node: one margin pass per GPU (up to four), then the
# comparisons, which are CPU work on the same node. Scores, reports and validation records
# follow the standard in scripts/operating_point_control.py (op-margin-v2).
#
#   PASSES="cfgA.yaml:adapter cfgB.yaml:adapter cfgA.yaml:base" \
#   COMPARE="cfgB.yaml:cfgA.yaml cfgA.yaml:base" \
#     scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/op_control.sbatch
#
# COMPARE entries are run:against (compare-arms) or cfg:base (the adapter against its base).
# VALIDATE="cfgA.yaml:adapter ..." checks the scorer against HuggingFace first (one entry per
# GPU); it is needed once per vLLM version, and reports only conform once a record passes.
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
read -r -a validations <<< "${VALIDATE:-}"
read -r -a passes <<< "${PASSES:-}"
read -r -a compares <<< "${COMPARE:-}"
[ $(( ${#validations[@]} + ${#passes[@]} + ${#compares[@]} )) -gt 0 ] \
  || { echo "FATAL: set VALIDATE, PASSES and/or COMPARE"; exit 2; }
[ ${#validations[@]} -le 4 ] && [ ${#passes[@]} -le 4 ] \
  || { echo "FATAL: at most four VALIDATE and four PASSES entries, one per GPU"; exit 2; }
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

# `operating_point_control.py <cmd>` for each cfg:which entry, one per GPU, all at once.
on_gpus() {
  local cmd=$1; shift
  local entries=("$@") pids=() fail=0 g log
  for g in "${!entries[@]}"; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/operating_point_control.py "$cmd" \
      --config "${entries[$g]%%:*}" --which "${entries[$g]##*:}" \
      > "$XW/logs/xw-opc-${SLURM_JOB_ID:-local}-$cmd-gpu$g.log" 2>&1 &
    pids+=($!)
  done
  for g in "${!pids[@]}"; do wait "${pids[$g]}" || fail=1; done
  for g in "${!entries[@]}"; do
    log="$XW/logs/xw-opc-${SLURM_JOB_ID:-local}-$cmd-gpu$g.log"
    echo "--- gpu$g $cmd ${entries[$g]}"
    { grep -E "wrote|skipping|passed|failed|Error|Traceback" "$log" || true; } | tail -n 3
  done
  return "$fail"
}

if [ ${#validations[@]} -gt 0 ]; then
  step "validate the scorer: ${validations[*]}"
  on_gpus validate "${validations[@]}" || { echo "FATAL: a validation failed -- see the gpu logs"; exit 1; }
fi
if [ ${#passes[@]} -gt 0 ]; then
  step "margin passes: ${passes[*]}"
  on_gpus margins "${passes[@]}" || { echo "FATAL: a margin pass failed -- see the gpu logs"; exit 1; }
fi

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
