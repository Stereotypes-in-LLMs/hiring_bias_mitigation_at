#!/bin/bash
# On-policy v2 negatives on batch-01's rows, in three stages that each fit the debug cap:
#
#   sample  merge the SFT adapter, check merged-vLLM vs HF+PEFT, sample the student (k per row,
#           with and without the attribute), select covert attribute-tipped mistakes
#   judge   the teacher judges them; candidates and compare/ (side by side with covert_flip);
#           then the GPU-free direction checks on compare/
#   check   D4 on compare/ with the SFT student and the base model (HF + PEFT, one GPU)
#
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy.sbatch sample
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy.sbatch judge
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/onpolicy.sbatch check
#
#SBATCH --job-name=xw-v2-onpolicy
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
CONFIG=configs/generation/negatives_v2_onpolicy_gh200.yaml
COMPARE=artifacts/semisynthetic-v2/batch-01-onpolicy/compare/pairs.parquet
STUDENT=Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | stage $STAGE | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

case "$STAGE" in
  sample)
    step "sample: merge, sanity check, student samples, selection"
    CUDA_VISIBLE_DEVICES=0 $PY scripts/generate_onpolicy_negatives.py --config "$CONFIG" --stage sample ;;
  judge)
    step "judge: teacher judgements, candidates, compare/"
    $PY scripts/generate_onpolicy_negatives.py --config "$CONFIG" --stage judge
    step "direction check, GPU-free part, on compare/"
    $PY scripts/check_preference_direction.py --pairs "$COMPARE" ;;
  check)
    step "D4 on compare/: SFT student and base model (HF + PEFT)"
    CUDA_VISIBLE_DEVICES=0 $PY scripts/check_preference_direction.py --pairs "$COMPARE" \
      --student "$STUDENT" --also-base --batch-size 4 ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac
echo "STAGE $STAGE OK"
