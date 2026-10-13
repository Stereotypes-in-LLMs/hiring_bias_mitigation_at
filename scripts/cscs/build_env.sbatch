#!/bin/bash
# Builds the locked Python environment on a GH200 node (aarch64) and proves it works there:
# versions, GPUs, the unit tests, and a two-prompt vLLM generation with the 9B student through
# the repo's own backend. Needs scripts/cscs/download.sbatch to have finished.
#
# debug partition: starts quickly, 30-minute cap. One node, one task, one submission.
#
#   scripts/cscs/submit.sh scripts/cscs/build_env.sbatch
#
#SBATCH --job-name=xw-env
#SBATCH --account=lp193
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
UV=$XW/bin/$(uname -m)/uv
cd "$REPO"
echo "host $(hostname) ($(uname -m)) | job ${SLURM_JOB_ID:-none} | $($UV --version)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

step "0/4 network from a GH200 node (PyPI and GitHub for wheels and Python)"
for url in https://pypi.org/simple/torch/ https://github.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$url" || true)
  echo "  $url -> $code"
  [[ $code =~ ^[23] ]] || { echo "FATAL: $url unreachable from a GPU node -- prefetch wheels on xfer instead"; exit 3; }
done

step "1/4 Python 3.12 environment from uv.lock (frozen)"
$UV sync --frozen --python 3.12 --extra vllm --extra dev
$PY - <<'PYEOF'
from importlib.metadata import version
pkgs = ["torch", "vllm", "transformers", "trl", "peft", "accelerate", "datasets",
        "huggingface-hub", "scikit-learn", "pandas", "pyarrow"]
print({p: version(p) for p in pkgs})
PYEOF

step "2/4 GPUs as torch sees them"
export HF_HUB_OFFLINE=1
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0), '| bf16', torch.cuda.is_bf16_supported())"

step "3/4 unit tests"
$PY -m pytest tests/ -q -x -p no:cacheprovider

step "4/4 vLLM smoke: Qwen3.5-9B through the repo's backend, two prompts, one GPU"
CUDA_VISIBLE_DEVICES=0 $PY scripts/cscs/vllm_smoke.py

echo "ENV OK"
