#!/bin/bash
# On-policy v2 negatives over the whole English pool, one node, three stages:
#
#   sample  prepare (pool rows, prompts, merged student), then one sampling process per GPU
#           over its share of the chunks, then collect if every chunk exists. A run cut by the
#           time limit loses at most one chunk per GPU: resubmit and it continues.
#   judge   collect (idempotent), the teacher judges the selected negatives and their chosen
#           side on all four GPUs, candidates + train/validation split, GPU-free checks
#   check   balance hire/reject per group inside each split (balanced/, CPU only), then the
#           direction check on the balanced pairs: D1-D6 on all, D4 on a 600-pair subsample
#           (the config's student and the model it was trained from, HF + PEFT: SFT and the
#           base, or a DPO checkpoint and SFT for an iterative round)
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch sample
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch judge
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch check
#
# Another config (e.g. the hire top-up) goes in through the environment; the output directory
# is read from it:
#   CONFIG=configs/generation/negatives_v2_onpolicy_hire_gh200.yaml scripts/cscs/submit.sh ...
#
# CHECK_PAIRS=candidates runs `check` on every judged pair, unbalanced, in $OUT/check_all/,
# for a run whose hire side is too thin to balance (e.g. an attribute-coverage run).
#
#SBATCH --job-name=xw-v2-onpolicy-full
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
STAGE=${1:?stage: sample | judge | check}
CONFIG=${CONFIG:-configs/generation/negatives_v2_onpolicy_full_gh200.yaml}
cd "$REPO"
OUT=$($PY -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['artifacts_dir'])" "$CONFIG")
STUDENT=$($PY -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['student_adapter'])" "$CONFIG")
echo "config $CONFIG | out $OUT | student $STUDENT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | stage $STAGE | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

case "$STAGE" in
  sample)
    step "prepare: pool rows, prompts, merged student"
    cost datagen prepare $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage prepare
    step "sample: one process per GPU"
    pids=(); fail=0
    for g in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$g cost datagen sample \
        $PY scripts/sample_student.py --config "$CONFIG" --shard $g --shards 4 \
        > "$XW/logs/xw-v2-onpolicy-full-${SLURM_JOB_ID:-local}-gpu$g.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    for g in 0 1 2 3; do echo "--- gpu$g"; grep -E "shard|Error|Traceback" "$XW/logs/xw-v2-onpolicy-full-${SLURM_JOB_ID:-local}-gpu$g.log" | tail -n 2; done
    [ "$fail" -eq 0 ] || { echo "FATAL: a sampling process failed -- see the gpu logs"; exit 1; }
    step "collect"
    cost datagen collect $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage collect ;;
  judge)
    step "collect (idempotent)"
    $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage collect > /dev/null
    step "judge: teacher on four GPUs"
    cost datagen judge $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage judge
    step "direction check, GPU-free part"
    cost datagen direction-check $PY scripts/check_preference_direction.py --batch "$OUT" ;;
  check)
    if [ "${CHECK_PAIRS:-balanced}" = candidates ]; then
      step "every judged pair, unbalanced, copied to check_all/"
      mkdir -p "$XW/$OUT/check_all"
      cp "$XW/$OUT/dpo_v2_candidates.parquet" "$XW/$OUT/report.json" "$XW/$OUT/check_all/"
      PAIRS="$XW/$OUT/check_all/dpo_v2_candidates.parquet"
    else
      step "balance: equal hire and reject per protected group inside each split"
      cost datagen balance $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage balance
      PAIRS="$OUT/balanced/dpo_v2_candidates.parquet"
    fi
    step "direction check on $PAIRS; D4 on a 600-pair subsample (HF + PEFT)"
    CUDA_VISIBLE_DEVICES=0 cost datagen d4-check $PY scripts/check_preference_direction.py \
      --pairs "$PAIRS" \
      --student "$STUDENT" --also-base --batch-size 4 --student-sample 600 ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "STAGE $STAGE OK"
