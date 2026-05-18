#!/usr/bin/env bash
set -euo pipefail

MODEL="${SGLANG_MODEL:-/root/sii-agent/Qwen3.5-9B}"
HOST="${SGLANG_HOST:-127.0.0.1}"
PORT="${SGLANG_PORT:-8004}"
TP="${SGLANG_TP:-4}"
CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-32768}"
MEM_FRACTION="${SGLANG_MEM_FRACTION_STATIC:-0.82}"
SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-Qwen3.5-9B}"
LOG_DIR="${SGLANG_LOG_DIR:-logs/services}"
LOG_FILE="${LOG_DIR}/sglang_qwen35_9b_${PORT}.log"
PYTHON_BIN="${SGLANG_PYTHON:-python}"

mkdir -p "${LOG_DIR}"

args=(
  -m sglang.launch_server
  --model-path "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --tp "${TP}"
  --context-length "${CONTEXT_LENGTH}"
  --mem-fraction-static "${MEM_FRACTION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --trust-remote-code
  --reasoning-parser "${SGLANG_REASONING_PARSER:-qwen3}"
  --tool-call-parser "${SGLANG_TOOL_CALL_PARSER:-hermes}"
  --attention-backend "${SGLANG_ATTENTION_BACKEND:-triton}"
  --prefill-attention-backend "${SGLANG_PREFILL_ATTENTION_BACKEND:-triton}"
  --decode-attention-backend "${SGLANG_DECODE_ATTENTION_BACKEND:-triton}"
  --mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND:-sdpa}"
)

if [[ -n "${SGLANG_LORA_PATH:-}" ]]; then
  args+=(
    --enable-lora
    --max-lora-rank "${SGLANG_MAX_LORA_RANK:-8}"
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --lora-paths "${SGLANG_LORA_NAME:-${SERVED_MODEL_NAME}}=${SGLANG_LORA_PATH}"
  )
fi

exec "${PYTHON_BIN}" "${args[@]}" "$@" 2>&1 | tee -a "${LOG_FILE}"
