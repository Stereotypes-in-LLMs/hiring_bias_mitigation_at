#!/bin/bash
# One frozen checkpoint, optional full margin pass, then its four contrasts.
# Reuses completed valid artifacts; same reference-selected pairs for all seeds.
#SBATCH --job-name=xw-poster-mechanism
#SBATCH --account=lp193
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out
set -euo pipefail
REPO=${REPO:-/ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at}
source "$REPO/scripts/cscs/env.sh"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$REPO/src:$XW/pylib-fla${PYTHONPATH:+:$PYTHONPATH}"
export FLA_DISABLE_BACKEND_DISPATCH=1
cd "$REPO"
SPEC=${POSTER_SPEC:?set POSTER_SPEC to a registered extra seed}
"$PY" scripts/check_mechanism_followup.py --spec "$SPEC" --check inputs
"$PY" scripts/check_mechanism_followup.py --spec "$SPEC" --check prepare-caches
if "$PY" scripts/check_mechanism_followup.py --spec "$SPEC" --check margins; then
  echo "Reusing completed full-grid margins for $SPEC"
else
  status=$?
  [ "$status" -eq 3 ] || exit "$status"
  CUDA_VISIBLE_DEVICES=0 cost patching "margins-$SPEC" \
    "$PY" scripts/interp/02_margins.py --spec "$SPEC" --dataset grid --device cuda:0
  "$PY" scripts/check_mechanism_followup.py --spec "$SPEC" --check margins
fi
contrasts=(mil_combat_civ_impl mil_combat_civ_expl rel_atheist_christian_impl gen_male_female_impl)
# Finish validation before starting any children so an invalid cached contrast
# cannot leave another contrast running after the parent exits.
pending_gpus=()
for gpu in "${!contrasts[@]}"; do
  contrast=${contrasts[$gpu]}
  if "$PY" scripts/check_mechanism_followup.py --spec "$SPEC" --check patch --contrast "$contrast"; then
    echo "Reusing completed patching for $SPEC/$contrast"
    continue
  else
    status=$?
    [ "$status" -eq 3 ] || exit "$status"
  fi
  pending_gpus+=("$gpu")
done
pids=()
for gpu in "${pending_gpus[@]}"; do
  contrast=${contrasts[$gpu]}
  CUDA_VISIBLE_DEVICES=$gpu cost patching "patch-$SPEC-$contrast" \
    "$PY" scripts/interp/10_patch.py --spec "$SPEC" --dataset grid --contrast "$contrast" \
    --select-on sft_v3 --n-pairs 64 --seed 42 --device cuda:0 \
    > "$XW/logs/xw-poster-mechanism-${SLURM_JOB_ID:-local}-$SPEC-$contrast.log" 2>&1 &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
[ "$fail" -eq 0 ] || { echo "FATAL: a patching task failed" >&2; exit 1; }
for contrast in "${contrasts[@]}"; do
  "$PY" scripts/check_mechanism_followup.py --spec "$SPEC" --check patch --contrast "$contrast"
done
"$PY" scripts/interp/20_distill_compare.py --spec "$SPEC" --ref sft_v3 \
  --contrast mil_combat_civ_impl --contrast mil_combat_civ_expl \
  --contrast rel_atheist_christian_impl --contrast gen_male_female_impl
mkdir -p reports/experiment_followup
"$PY" - "$SPEC" <<'PY'
import json, sys
from pathlib import Path
spec = sys.argv[1]
Path(f"reports/experiment_followup/mechanism_{spec}.json").write_text(
    json.dumps({"spec": spec, "completed": True, "selection_reference": "sft_v3",
                "selection_seed": 42, "requested_pairs": 64,
                "gender_actual_pairs": 45, "contrasts": 4}, indent=2)+"\n")
PY
echo "STAGE mechanism follow-up OK: $SPEC"
