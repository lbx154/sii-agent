#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-18877}"
LOG_DIR="${VLLM_LOG_DIR:-logs/vllm}"
LOG_FILE="${LOG_DIR}/qwen35.log"
VLLM_BIN="${VLLM_BIN:-vllm}"

mkdir -p "${LOG_DIR}"

exec "${VLLM_BIN}" serve "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-8192}" \
  --dtype "${VLLM_DTYPE:-bfloat16}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}" \
  --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-64}" \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser "${VLLM_TOOL_CALL_PARSER:-hermes}" \
  --reasoning-parser "${VLLM_REASONING_PARSER:-qwen3}" \
  --trust-remote-code \
  "$@" 2>&1 | tee -a "${LOG_FILE}"
