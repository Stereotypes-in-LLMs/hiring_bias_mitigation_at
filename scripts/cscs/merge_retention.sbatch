#!/bin/bash
# Measures how much of each adapter's update its bf16 merge kept, for adapters merged before
# mitigation/merge.py recorded it, then refreshes those runs' reference rows and prints their
# verdicts (docs/EVAL_STANDARD.md). The weights already merged are not touched. CPU work: the
# base is loaded in bf16 on the node's CPU, one config after the other.
#
#   CONFIGS="configs/audit/qwen3.5-9b_en_sft_en_only_gh200.yaml configs/audit/qwen3.5-9b_en_dpo_v2neg_r2em_step100_gh200.yaml" \
#     scripts/cscs/submit.sh scripts/cscs/merge_retention.sbatch
#
#SBATCH --job-name=xw-merge-retention
#SBATCH --account=lp193
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
# An env.sh without `cost` (main before the recorder, an older checkout): run the step
# unrecorded rather than fail.
type cost > /dev/null 2>&1 || cost() { shift 2; "$@"; }
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
read -r -a configs <<< "${CONFIGS:?set CONFIGS to the audit configs of the adapters to measure}"
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"

step "measure the merges: ${configs[*]}"
cost eval merge-retention $PY scripts/merge_adapter.py --config "${configs[@]}" --retention-only

step "refresh the reference rows"
$PY scripts/reference_metrics.py --config "${configs[@]}" --show
echo "STAGE merge-retention OK"
