#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_ROOT}/third_party/slime"
source "${SCRIPT_DIR}/qwen3.5-9B.sh"

: "${HF_CHECKPOINT:=/root/sii-agent/Qwen3.5-9B}"
: "${SAVE_PATH:=/root/sii-agent/Qwen3.5-9B_torch_dist}"
: "${CONVERT_CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${CONVERT_GPUS:=4}"
: "${MEGATRON_PATH:=${REPO_ROOT}/third_party/Megatron-LM}"

if [ ! -d "${HF_CHECKPOINT}" ]; then
  echo "Missing HF checkpoint: ${HF_CHECKPOINT}" >&2
  exit 1
fi
if [ ! -d "${SLIME_DIR}" ]; then
  echo "Missing slime checkout: ${SLIME_DIR}" >&2
  exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
  echo "Missing Megatron-LM at ${MEGATRON_PATH}; finish slime build_conda.sh first." >&2
  exit 1
fi

cd "${SLIME_DIR}"
export CUDA_VISIBLE_DEVICES="${CONVERT_CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="${MEGATRON_PATH}:${SLIME_DIR}:${REPO_ROOT}:${PYTHONPATH:-}"

torchrun --nproc-per-node "${CONVERT_GPUS}" "${REPO_ROOT}/scripts/slime/convert_hf_to_torch_dist_qwen35.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --save "${SAVE_PATH}"
