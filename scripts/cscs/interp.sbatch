#!/bin/bash
# The activation-patching pipeline (reports/interp/README.md) on one GH200 node: up to four
# tasks at once, one per GPU, each in its own log.
#
#   TASKS="margins:base margins:sft_v3 margins:distill43 margins:distill42" \
#   MARGIN_ARGS="--resid" scripts/cscs/submit.sh -t 03:00:00 scripts/cscs/interp.sbatch
#   TASKS="patch:sft_v3:mil_combat_civ_impl:sft_v3 patch:distill43:mil_combat_civ_impl:sft_v3" \
#     scripts/cscs/submit.sh -t 03:00:00 scripts/cscs/interp.sbatch
#
# margins:SPEC runs 02_margins.py on the grid; patch:SPEC:CONTRAST[:SELECT_ON] runs 10_patch.py
# on the grid, on pairs selected by SELECT_ON's effect (default base). Both resume from their
# finished parts, so a job cut by the time limit continues when it is submitted again.
# The GDN kernels (fla) come from $XW/pylib-fla, not the training venv, so training keeps its
# kernels; io.kernel_info records which path each artifact ran on.
#
#SBATCH --job-name=xw-interp
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=03:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src:$XW/pylib-fla${PYTHONPATH:+:$PYTHONPATH}"
read -r -a tasks <<< "${TASKS:-}"
[ ${#tasks[@]} -ge 1 ] && [ ${#tasks[@]} -le 4 ] \
  || { echo "FATAL: set TASKS to one to four entries, one per GPU"; exit 2; }
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | branch $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo n/a)"
$PY -c "from hiring_bias_mitigation.interp.io import kernel_info; print('kernels', kernel_info())"

step "interp: ${tasks[*]}"
pids=()
for g in "${!tasks[@]}"; do
  IFS=: read -r kind spec contrast select_on <<< "${tasks[$g]}"
  log="$XW/logs/xw-interp-${SLURM_JOB_ID:-local}-gpu$g-$kind-$spec${contrast:+-$contrast}.log"
  case $kind in
    margins) cmd=($PY scripts/interp/02_margins.py --spec "$spec" --dataset grid
                  --device cuda:0 ${MARGIN_ARGS:-}) ;;
    patch)   cmd=($PY scripts/interp/10_patch.py --spec "$spec" --dataset grid
                  --contrast "$contrast" --select-on "${select_on:-base}" --device cuda:0
                  ${PATCH_ARGS:-}) ;;
    *) echo "FATAL: unknown task kind $kind"; exit 2 ;;
  esac
  CUDA_VISIBLE_DEVICES=$g cost patching "$kind-$spec${contrast:+-$contrast}" "${cmd[@]}" \
    > "$log" 2>&1 &
  pids+=($!)
done
fail=0
for g in "${!pids[@]}"; do wait "${pids[$g]}" || fail=1; done
for g in "${!tasks[@]}"; do
  echo "--- gpu$g ${tasks[$g]}"
  { grep -E "wrote|summaries|agreement|hire_rate|part [0-9]+/|patches in|Error|Traceback" \
      "$XW"/logs/xw-interp-"${SLURM_JOB_ID:-local}"-gpu$g-*.log || true; } | tail -n 6
done
[ "$fail" -eq 0 ] || { echo "FATAL: a task failed -- see $XW/logs/xw-interp-${SLURM_JOB_ID:-local}-gpu*.log"; exit 1; }
echo "STAGE interp OK"
