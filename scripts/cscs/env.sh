# Shared environment for the CSCS Daint jobs. Sourced by every scripts/cscs/*.sbatch, never
# run on its own. Everything a job downloads, caches or writes lives under $XW on scratch.
export XW=${XW:-/ritom/scratch/cscs/zhang/xw}
export REPO=${REPO:-$XW/hiring_bias_mitigation_at}

# The repo's own output root: configs keep short relative paths (artifacts/..., outputs/...),
# and utils/config.py refuses to write unless $HBM_OUTPUT_ROOT/.hbm-models-root exists.
export HBM_OUTPUT_ROOT=$XW

# Every cache off $HOME (50 GB quota) and onto scratch.
export HF_HOME=$XW/hf-cache
export UV_CACHE_DIR=$XW/uv-cache
export UV_PYTHON_INSTALL_DIR=$XW/uv-python
export UV_TOOL_DIR=$XW/uv-tools
export UV_TOOL_BIN_DIR=$XW/bin
export UV_PROJECT_ENVIRONMENT=$XW/venv
export XDG_CACHE_HOME=$XW/.cache
export TRITON_CACHE_DIR=$XW/.cache/triton
export VLLM_CACHE_ROOT=$XW/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$XW/.cache/inductor
export WANDB_DIR=$XW/wandb

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Daint GH200 nodes have no system CUDA toolkit (/usr/local/cuda), so nothing may JIT-compile
# FlashInfer kernels at startup. vLLM 0.31 does that twice on Hopper: its top-k/top-p sampler
# (switched off here, falling back to vLLM's own sampler) and the Qwen3.5 linear-attention
# prefill kernel (switched to Triton per config: engine_kwargs.gdn_prefill_backend).
# Found by the first build_env run (job 5043998): "Could not find nvcc".
export VLLM_USE_FLASHINFER_SAMPLER=0
# FlashInfer all-reduce cannot work here either (no nvcc for its JIT, no NVSwitch multicast).
# Job 5045685 tried it, failed fast and fell back; job 5051986 hung 20+ minutes inside the
# fallback at CUDA-graph capture. Off, tensor parallel uses the custom all-reduce and NCCL.
export VLLM_ALLREDUCE_USE_FLASHINFER=0
# W&B: the key sits in a file only this account can read (mode 600), not in ~/.netrc, which
# every user of the shared account would inherit. Without it, training logs offline under
# $WANDB_DIR, and train_dpo.sbatch uploads those runs at the start of the next segment.
export WANDB_DIR=$XW/wandb
if [ -r "$XW/wandb/wandb/wandb_apikey" ]; then
  WANDB_API_KEY=$(tr -d '\n\r ' < "$XW/wandb/wandb/wandb_apikey")
  export WANDB_API_KEY
fi
[ -n "${WANDB_API_KEY:-}" ] || export WANDB_MODE=${WANDB_MODE:-offline}
# Uploads offline W&B runs that were never synced; wandb sync takes a resumed run's files in
# start order, so earlier segments land before the next one continues online. No route out of
# the node: says so and leaves them for the next job.
wandb_upload_offline() {
  if [ -z "${WANDB_API_KEY:-}" ] || [ "${WANDB_MODE:-online}" = offline ]; then
    echo "W&B upload skipped: no key or WANDB_MODE=offline"
    return 0
  fi
  local offline
  shopt -s nullglob; offline=("$WANDB_DIR"/wandb/offline-run-*); shopt -u nullglob
  [ ${#offline[@]} -gt 0 ] || return 0
  if ! curl -s -m 10 -o /dev/null https://api.wandb.ai; then
    echo "no route to api.wandb.ai from $(hostname): offline runs wait for a later job"
    return 0
  fi
  timeout 300 "$XW/venv/bin/wandb" sync --yes "${offline[@]}" \
    || echo "wandb sync failed; the runs stay offline for the next job"
}

# Usable GPU memory is 95.6 GiB (97,871 MiB) per GH200, despite "120GB" in the device name;
# vLLM's gpu_memory_utilization is a fraction of that, and anything near 96 GB OOMs.

PY=$XW/venv/bin/python

# Pinned to what uv.lock was written with, so every node resolves the same environment.
export UV_VERSION=0.10.9
export HF_HUB_VERSION=1.33.0

step() { echo; echo "=== $(date '+%F %T') $*"; }

# `cost <stage> <label> <command...>` runs the command as one recorded step: wall time, the
# GPUs its processes actually used, their energy, the job, the commit, plus what the code adds
# (tokens, trained parameters). One JSON per step under $HBM_COST_DIR; scripts/cost_report.py
# builds reports/cost/ from them. Stages: datagen | train | eval | patching | other.
# HBM_COST_ARM=<arm> names the arm a step's cost belongs to. If the recorder cannot be
# imported, the command runs unrecorded: bookkeeping never costs a run.
export HBM_COST_DIR=${HBM_COST_DIR:-$XW/cost/records}
cost() {
  local stage=$1 label=$2
  shift 2
  if "$PY" -c "import hiring_bias_mitigation.utils.cost" 2>/dev/null; then
    "$PY" -m hiring_bias_mitigation.utils.cost run --stage "$stage" --label "$label" -- "$@"
  else
    echo "cost: recorder not importable here; running $stage/$label unrecorded" >&2
    "$@"
  fi
}
