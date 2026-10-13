#!/bin/bash
# Submits one scripts/cscs job from a Daint login node:
#
#   scripts/cscs/submit.sh scripts/cscs/download.sbatch
#   scripts/cscs/submit.sh -p debug -t 00:30:00 scripts/cscs/small_batch.sbatch generate
#
# Slurm opens a job's --output file before the job starts and does not create directories,
# so $XW/logs has to exist at submission time; on a fresh $XW a bare `sbatch` fails with no
# log at all. Creating it and calling sbatch is all this does -- nothing heavy runs here.
set -euo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
mkdir -p "$XW/logs"
cd "$REPO"
exec sbatch "$@"
