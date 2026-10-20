#!/bin/bash
# Part B of the GTC poster (docs/THROUGHPUT_B.md): the counterfactual audit's generation timed
# on one GH200 node under four conditions -- vLLM prefix caching on/off x prompts submitted in
# prefix order/shuffled -- one condition per GPU, in parallel; then a second repeat with every
# condition moved two GPUs over, so no condition is timed on one GPU only. Each run is one
# `cost eval throughput-<condition>` step (wall time, GPU, energy, tokens, prefix-cache
# counters). Nothing is scored: the runs write outputs/throughput/raw/, never eval/results/.
#
#   REPO=$XW/hiring_bias_mitigation_at-throughput \
#     scripts/cscs/submit.sh scripts/cscs/throughput_b.sbatch
#
# Then, on the login node: scripts/cost_report.py, scripts/throughput_b.py speedup.
#
#SBATCH --job-name=xw-throughput-b
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source "${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}/scripts/cscs/env.sh"
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
# The venv's editable install points at the main checkout; this job runs $REPO's code.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
# ...but `cost` also runs a step unrecorded when the recorder does not import, and a timing
# run without its record measures nothing: stop instead.
"$PY" -c "import hiring_bias_mitigation.utils.cost" \
  || { echo "FATAL: the cost recorder does not import from $REPO/src"; exit 3; }
cd "$REPO"
[ -f scripts/throughput_b.py ] \
  || { echo "FATAL: $REPO has no scripts/throughput_b.py; submit with REPO set to its checkout"; exit 2; }
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | repo $REPO | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | uncommitted $(git status --porcelain -- src scripts configs 2>/dev/null | wc -l) files"
# The code that ran, when it is not committed: every changed or new file and its hash.
git status --porcelain --untracked-files=all -- src scripts configs tests 2>/dev/null | while read -r _ f; do
  [ -f "$f" ] && echo "code $(sha256sum "$f" | cut -c1-16) $f"
done
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit,temperature.gpu --format=csv

conditions=(cacheon-prefix cacheon-shuffle cacheoff-prefix cacheoff-shuffle)
REPEATS=${REPEATS:-2}
for r in $(seq 1 "$REPEATS"); do
  step "repeat r$r"
  pids=(); fail=0
  for i in "${!conditions[@]}"; do
    c=${conditions[$i]}
    g=$(( (i + 2 * (r - 1)) % 4 ))
    log="$XW/logs/xw-throughput-b-${SLURM_JOB_ID:-local}-$c-r$r.log"
    echo "r$r $c on GPU $g -> $log"
    CUDA_VISIBLE_DEVICES=$g cost eval "throughput-$c" \
      "$PY" scripts/throughput_b.py run --config "configs/throughput/qwen3.5-9b_en_$c.yaml" \
      --repeat "r$r" > "$log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  for c in "${conditions[@]}"; do
    echo "--- $c r$r"
    { grep -E "Traceback|Error|prompts, wall|cost: eval" \
        "$XW/logs/xw-throughput-b-${SLURM_JOB_ID:-local}-$c-r$r.log" || true; } | tail -n 4
  done
  [ "$fail" -eq 0 ] || { echo "FATAL: a timing run failed in repeat r$r -- see its log"; exit 1; }
done

step "greedy outputs across conditions"
"$PY" scripts/throughput_b.py compare
echo "STAGE throughput-b OK"
