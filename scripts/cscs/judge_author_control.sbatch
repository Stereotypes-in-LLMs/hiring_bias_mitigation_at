#!/bin/bash
# The judge-author control (docs/JUDGE_AUTHOR_CONTROL.md): the v2 judge on the SFT student's
# correct samples from the judged rows of onpolicy-full and onpolicy-hire32, plus a retest of
# 400 hire rows' October judgements per round. Select and summarize are CPU; the judge is the
# 122B teacher on all four GPUs (tensor parallel 4), which takes about 5 minutes to load.
#
# Debug first (64 seeded rows per round, separate files under judge_control-debug/):
#   TAG=debug LIMIT=64 RETEST_N=16 scripts/cscs/submit.sh -p debug -t 00:30:00 \
#     scripts/cscs/judge_author_control.sbatch
# Then the full run:
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/judge_author_control.sbatch
#
#SBATCH --job-name=xw-judge-author
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source "${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}/scripts/cscs/env.sh"
[ -x "$PY" ] || { echo "FATAL: no venv at $XW/venv -- run scripts/cscs/build_env.sbatch first"; exit 3; }
export HF_HUB_OFFLINE=1
cd "$REPO"
TAG=${TAG:-}
ARGS=(--tag "$TAG")
[ -n "${LIMIT:-}" ] && ARGS+=(--limit "$LIMIT")
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | tag ${TAG:-none} | limit ${LIMIT:-all} | commit $(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
git status --porcelain --untracked-files=all -- scripts src reports/judge_author_control | sed 's/^/uncommitted: /'
[ -f reports/judge_author_control/PREREG.md ] && echo "prereg sha256 $(sha256sum reports/judge_author_control/PREREG.md | cut -c1-16)"

step "tests"
$PY -m pytest -q tests/test_judge_author_control.py tests/test_onpolicy.py tests/test_negatives_v2.py
step "select: first correct filtered sample per judged row"
$PY scripts/judge_author_control.py --stage select "${ARGS[@]}"
step "judge: control samples + retest, teacher on four GPUs"
$PY scripts/judge_author_control.py --stage judge "${ARGS[@]}" --retest-n "${RETEST_N:-400}"
step "summarize"
$PY scripts/judge_author_control.py --stage summarize "${ARGS[@]}"
echo "STAGE judge-author-control OK"
