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
export WANDB_MODE=${WANDB_MODE:-offline}

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

# Usable GPU memory is 95.6 GiB (97,871 MiB) per GH200, despite "120GB" in the device name;
# vLLM's gpu_memory_utilization is a fraction of that, and anything near 96 GB OOMs.

PY=$XW/venv/bin/python

# Pinned to what uv.lock was written with, so every node resolves the same environment.
export UV_VERSION=0.10.9
export HF_HUB_VERSION=1.33.0

step() { echo; echo "=== $(date '+%F %T') $*"; }
