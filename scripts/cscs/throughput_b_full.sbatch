#!/bin/bash
# Part B, round 2 (docs/THROUGHPUT_B.md): the whole audit of DPO_V3_4880 (450 pairs, 31,050
# prompts, merged checkpoint-305) timed on one GH200 node under four conditions -- the engine
# every audit used (prefix cache on, built order, stat logging off), the same with stat logging
# on, prefix order, and round 1's naive reference (cache off, shuffled). A seeded 4 x 4 Latin
# square (scripts/throughput_b.py schedule): each GPU runs all four conditions one after another,
# and each condition runs once in each position, so every comparison has four same-GPU repeats
# and no condition always runs first. A warm-up pass first (one condition per GPU, 4 pairs,
# nothing kept) fills the node's file and compile caches. Each run is one
# `cost eval throughput-full-<condition>` step. Nothing is scored: the runs write
# outputs/throughput/full/raw/, never eval/results/.
#
#   REPO=$XW/hiring_bias_mitigation_at-throughput-full \
#     scripts/cscs/submit.sh scripts/cscs/throughput_b_full.sbatch
#
# Then, on the login node: scripts/cost_report.py, and scripts/throughput_b.py speedup / compare /
# quality with --round full.
#
#SBATCH --job-name=xw-throughput-full
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:45:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source "${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}/scripts/cscs/env.sh"
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
# The venv's editable install points at the main checkout; this job runs $REPO's code.
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
# A timing run without its cost record measures nothing: stop instead of running unrecorded.
type cost > /dev/null 2>&1 || { echo "FATAL: env.sh has no cost recorder"; exit 3; }
"$PY" -c "import hiring_bias_mitigation.utils.cost" \
  || { echo "FATAL: the cost recorder does not import from $REPO/src"; exit 3; }
cd "$REPO"
[ -f configs/throughput/full/qwen3.5-9b_en_dpo_v3_4880_cacheon-prefix.yaml ] \
  || { echo "FATAL: $REPO has no configs/throughput/full/; submit with REPO set to its checkout"; exit 2; }
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | repo $REPO | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a) | uncommitted $(git status --porcelain -- src scripts configs 2>/dev/null | wc -l) files"
# The code that ran, when it is not committed: every changed or new file and its hash.
git status --porcelain --untracked-files=all -- src scripts configs tests 2>/dev/null | while read -r _ f; do
  [ -f "$f" ] && echo "code $(sha256sum "$f" | cut -c1-16) $f"
done
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit,temperature.gpu --format=csv
LOGS="$XW/logs/xw-throughput-full-${SLURM_JOB_ID:-local}"
SEED=${SEED:-2027}

step "warm-up: one condition per GPU, ${SLURM_JOB_ID:-local}, nothing kept"
mapfile -t plan < <("$PY" scripts/throughput_b.py schedule --round full --seed "$SEED")
[ "${#plan[@]}" -eq 16 ] || { echo "FATAL: the schedule has ${#plan[@]} runs, not 16"; exit 2; }
printf '%s\n' "${plan[@]}" | sed 's/^/schedule /'
pids=(); fail=0
for line in "${plan[@]}"; do
  read -r g _ pos _ cfg <<< "$line"
  [ "$pos" = 0 ] || continue
  # Each GPU warms the condition it runs first, so all four engines' compile caches are filled.
  CUDA_VISIBLE_DEVICES=$g cost eval throughput-full-warmup \
    "$PY" scripts/throughput_b.py run --warmup --repeat r0 --config "$cfg" \
    > "$LOGS-warmup-g$g.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "FATAL: a warm-up run failed -- see $LOGS-warmup-g*.log"; exit 1; }

step "timed runs: every GPU runs its row of the square in order"
pids=()
for g in 0 1 2 3; do
  (
    set -e
    for line in "${plan[@]}"; do
      read -r gpu repeat pos c cfg <<< "$line"
      [ "$gpu" = "$g" ] || continue
      echo "$(date '+%T') GPU $g position $pos $c $repeat -> $LOGS-$c-$repeat.log"
      CUDA_VISIBLE_DEVICES=$g cost eval "throughput-full-$c" \
        "$PY" scripts/throughput_b.py run --config "$cfg" --repeat "$repeat" \
        > "$LOGS-$c-$repeat.log" 2>&1
    done
  ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
for line in "${plan[@]}"; do
  read -r _ repeat _ c _ <<< "$line"
  echo "--- $c $repeat"
  { grep -E "Traceback|Error|prompts, wall|cost: eval" "$LOGS-$c-$repeat.log" || true; } | tail -n 4
done
[ "$fail" -eq 0 ] || { echo "FATAL: a timing run failed -- see its log"; exit 1; }
echo "STAGE throughput-full OK"
