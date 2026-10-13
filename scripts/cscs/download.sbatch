#!/bin/bash
# Fetches everything architecture-neutral the v2 small batch needs into $XW: uv binaries for
# both architectures, the teacher and student models, the published SFT adapter, and v1's
# English data. The Python environment is NOT built here: xfer nodes are x86_64 and the GPU
# nodes are aarch64, so the venv is built on a GH200 node by scripts/cscs/build_env.sbatch.
#
# Runs on the xfer partition (billing weight 0, shared node) -- never on a login node. One
# node, one task, one submission. Safe to resubmit: every step skips what already exists.
#
#   scripts/cscs/submit.sh scripts/cscs/download.sbatch
#
#SBATCH --job-name=xw-download
#SBATCH --account=lp193
#SBATCH --partition=xfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/ritom/scratch/cscs/zhang/xw/logs/%x-%j.out

set -euo pipefail
source /ritom/scratch/cscs/zhang/xw/hiring_bias_mitigation_at/scripts/cscs/env.sh
mkdir -p "$XW"/{logs,bin,tmp,hf-cache,uv-cache,uv-python,.cache,hub} \
         "$XW/artifacts/semisynthetic-v1/raw"
echo "host $(hostname) ($(uname -m)) | job ${SLURM_JOB_ID:-none} | XW=$XW"

step "0/4 network from this node"
for url in https://huggingface.co https://pypi.org/simple/uv/ https://github.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$url" || true)
  echo "  $url -> $code"
  [[ $code =~ ^[23] ]] || { echo "FATAL: $url unreachable from $(hostname)"; exit 3; }
done

step "1/4 uv $UV_VERSION for x86_64 and aarch64 (sha256 from the GitHub release)"
declare -A SHA=(
  [x86_64]=433e56874739e92c7cfd661ba9e5f287b376ca612c08c8194a41a98a13158aea
  [aarch64]=05b0d3087e913ebe11756365a90dd47c05d6728752fdbe129ad4c3ccd769826d
)
for arch in x86_64 aarch64; do
  if [ ! -x "$XW/bin/$arch/uv" ]; then
    tgz="uv-$arch-unknown-linux-musl.tar.gz"
    curl -fsSL -o "$XW/tmp/$tgz" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/$tgz"
    echo "${SHA[$arch]}  $XW/tmp/$tgz" | sha256sum -c -
    tar -xzf "$XW/tmp/$tgz" -C "$XW/tmp"
    mkdir -p "$XW/bin/$arch"
    mv "$XW/tmp/uv-$arch-unknown-linux-musl/uv" "$XW/tmp/uv-$arch-unknown-linux-musl/uvx" "$XW/bin/$arch/"
  fi
done
"$XW/bin/$(uname -m)/uv" --version
HF=("$XW/bin/$(uname -m)/uvx" --python 3.12 --from "huggingface_hub==$HF_HUB_VERSION" hf)

step "2/4 models into \$HF_HOME"
"${HF[@]}" download Qwen/Qwen3.5-9B
"${HF[@]}" download Stereotypes-in-LLMs/qwen3.5-9b-hiring-debias-sft-en
"${HF[@]}" download Qwen/Qwen3.5-122B-A10B-GPTQ-Int4

step "3/4 v1 English data into artifacts/semisynthetic-v1"
HUB=$XW/hub/hiring-bias-mitigation-synthetic-data
V1=$XW/artifacts/semisynthetic-v1
"${HF[@]}" download Stereotypes-in-LLMs/hiring-bias-mitigation-synthetic-data --repo-type dataset \
  --local-dir "$HUB" \
  data/raw/en_invariant.parquet data/raw/en_reference.parquet \
  data/sft_validation.parquet data/dpo_validation.parquet generation_report.json
cp -f "$HUB"/data/raw/en_invariant.parquet "$HUB"/data/raw/en_reference.parquet "$V1/raw/"
cp -f "$HUB"/data/sft_validation.parquet "$HUB"/data/dpo_validation.parquet \
      "$HUB"/generation_report.json "$V1/"
touch "$XW/.hbm-models-root"

step "4/4 summary"
du -sh "$XW"/hf-cache/hub/* "$V1" "$XW/bin" 2>/dev/null
ls -la "$V1" "$V1/raw"
echo "DOWNLOAD OK"
