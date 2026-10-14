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
#           (SFT student and base model, HF + PEFT)
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch sample
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch judge
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy_full.sbatch check
#
# Another config (e.g. the hire top-up) goes in through the environment; the output directory
# is read from it:
#   CONFIG=configs/generation/negatives_v2_onpolicy_hire_gh200.yaml scripts/cscs/submit.sh ...
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
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
STAGE=${1:?stage: sample | judge | check}
CONFIG=${CONFIG:-configs/generation/negatives_v2_onpolicy_full_gh200.yaml}
STUDENT=Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en
cd "$REPO"
OUT=$($PY -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['artifacts_dir'])" "$CONFIG")
echo "config $CONFIG | out $OUT"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | stage $STAGE | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

case "$STAGE" in
  sample)
    step "prepare: pool rows, prompts, merged student"
    $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage prepare
    step "sample: one process per GPU"
    pids=(); fail=0
    for g in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$g $PY scripts/sample_student.py --config "$CONFIG" --shard $g --shards 4 \
        > "$XW/logs/xw-v2-onpolicy-full-${SLURM_JOB_ID:-local}-gpu$g.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || fail=1; done
    for g in 0 1 2 3; do echo "--- gpu$g"; grep -E "shard|Error|Traceback" "$XW/logs/xw-v2-onpolicy-full-${SLURM_JOB_ID:-local}-gpu$g.log" | tail -n 2; done
    [ "$fail" -eq 0 ] || { echo "FATAL: a sampling process failed -- see the gpu logs"; exit 1; }
    step "collect"
    $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage collect ;;
  judge)
    step "collect (idempotent)"
    $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage collect > /dev/null
    step "judge: teacher on four GPUs"
    $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage judge
    step "direction check, GPU-free part"
    $PY scripts/check_preference_direction.py --batch "$OUT" ;;
  check)
    step "balance: equal hire and reject per protected group inside each split"
    $PY scripts/generate_onpolicy_full.py --config "$CONFIG" --stage balance
    step "direction check on the balanced pairs; D4 on a 600-pair subsample (HF + PEFT)"
    CUDA_VISIBLE_DEVICES=0 $PY scripts/check_preference_direction.py \
      --pairs "$OUT/balanced/dpo_v2_candidates.parquet" \
      --student "$STUDENT" --also-base --batch-size 4 --student-sample 600 ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "STAGE $STAGE OK"
