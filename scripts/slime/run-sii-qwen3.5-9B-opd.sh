#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_ROOT}/third_party/slime"
source "${SCRIPT_DIR}/qwen3.5-9B.sh"

for ENV_FILE in "${REPO_ROOT}/.env" "/root/harness-sii-browser-service/.env"; do
  if [ -f "${ENV_FILE}" ]; then
    set -a
    # shellcheck source=/dev/null
    source "${ENV_FILE}"
    set +a
  fi
done

if [ -x /root/myslime_env/bin/python ]; then
  : "${SLIME_PYTHON:=/root/myslime_env/bin/python}"
else
  : "${SLIME_PYTHON:=python}"
fi
if [[ "${SLIME_PYTHON}" == */* ]]; then
  SLIME_BIN_DIR="$(cd -- "$(dirname -- "${SLIME_PYTHON}")" &>/dev/null && pwd)"
  export PATH="${SLIME_BIN_DIR}:${PATH}"
  : "${RAY_BIN:=${SLIME_BIN_DIR}/ray}"
else
  : "${RAY_BIN:=ray}"
fi

: "${STUDENT_HF_CHECKPOINT:=/root/sii-agent/Qwen3.5-9B}"
: "${TEACHER_HF_CHECKPOINT:=/root/sii-agent/Qwen3.6-27B}"
: "${STUDENT_TORCH_DIST:=/root/sii-agent/Qwen3.5-9B_torch_dist}"
: "${SLIME_SAVE:=/root/sii-agent/saves/qwen35-9b/slime_agent_opd_browser}"
: "${PROMPT_DATA:=/root/sii-agent/data/slime/sii_agent_browser_opd_train.jsonl}"
: "${TEACHER_CUDA_VISIBLE_DEVICES:=6,7}"
: "${START_VISION_SERVER:=0}"
: "${VISION_CUDA_VISIBLE_DEVICES:=5}"
: "${VISION_TP:=1}"
: "${VISION_CONTEXT_LENGTH:=32768}"
: "${VISION_MEM_FRACTION_STATIC:=0.70}"
: "${VISION_PORT:=8004}"
if [[ "${START_VISION_SERVER}" =~ ^(1|true|True|yes|YES)$ ]]; then
  : "${STUDENT_CUDA_VISIBLE_DEVICES:=0,1,2,3,4}"
else
  : "${STUDENT_CUDA_VISIBLE_DEVICES:=0,1,2,3,4,5}"
fi
: "${TEACHER_PORT:=13141}"
: "${REUSE_TEACHER:=0}"
: "${VALIDATE_TEACHER_MODEL:=1}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${REUSE_RAY:=0}"
: "${RAY_HOST:=127.0.0.1}"
: "${RAY_GCS_PORT:=6379}"
: "${RAY_DASHBOARD_PORT:=8265}"
: "${RAY_DASHBOARD_HOST:=0.0.0.0}"
: "${RAY_CLIENT_SERVER_PORT:=10001}"
: "${RAY_MIN_WORKER_PORT:=10002}"
: "${RAY_MAX_WORKER_PORT:=19999}"
: "${RAY_OBJECT_MANAGER_PORT:=}"
: "${RAY_NODE_MANAGER_PORT:=}"
: "${RAY_DASHBOARD_AGENT_LISTEN_PORT:=}"
: "${RAY_DASHBOARD_AGENT_GRPC_PORT:=}"
: "${RAY_RUNTIME_ENV_AGENT_PORT:=}"
: "${RAY_METRICS_EXPORT_PORT:=}"
: "${RAY_TEMP_DIR:=/tmp/ray}"
: "${RAY_ADDRESS:=${RAY_HOST}:${RAY_GCS_PORT}}"
: "${RAY_JOB_ADDRESS:=http://${RAY_HOST}:${RAY_DASHBOARD_PORT}}"
: "${SII_SLIME_TRAIN_STEPS:=26}"
: "${SII_SLIME_MAX_STEPS:=26}"
: "${SII_SLIME_MAX_TURN_TOKENS:=8192}"
: "${SII_SLIME_MAX_OBSERVATION_CHARS:=4000}"
: "${SII_SLIME_MAX_REPEATS:=1}"
: "${SII_SLIME_USE_TASK_REWARD:=1}"
: "${SII_SEARCH_SIMILARITY_THRESHOLD:=0.82}"
: "${SII_SEARCH_MAX_SIMILAR_REPEATS:=1}"
: "${SII_SEARCH_MAX_CALLS_PER_TOOL:=8}"
: "${SII_SEARCH_MAX_CALLS_TOTAL:=16}"
: "${SII_TEACHER_RM_CONCURRENCY:=2}"
: "${SII_TEACHER_RM_RETRIES:=5}"
: "${SII_TEACHER_RM_TOTAL_TIMEOUT:=1800}"
: "${SII_TEACHER_RM_CONNECT_TIMEOUT:=300}"
: "${SII_TEACHER_RM_READ_TIMEOUT:=1800}"
: "${SII_TRAIN_CORRECT_ONLY:=0}"
: "${SII_TRAIN_DROP_TRUNCATED:=1}"
: "${SII_TRAIN_MIXED_GROUPS_ONLY:=1}"
: "${SII_TRAIN_REQUIRE_FINAL_ANSWER:=1}"
: "${SII_TRAIN_REQUIRE_EVIDENCE_OPEN:=1}"
: "${SII_TRAIN_MIN_SEARCH_CALLS_FOR_OPEN:=4}"
: "${SII_TRAIN_MAX_TOTAL_TOKENS:=30000}"
: "${SII_TRAIN_MAX_RESPONSE_TOKENS:=20000}"
: "${SII_TRAIN_MAX_TOOL_CALLS:=20}"
: "${SII_DISABLE_VISION_TOOLS_WHEN_UNAVAILABLE:=1}"
: "${SII_VISION_USE_ROLLOUT_ROUTER:=1}"
: "${SEARCH_PROXY_URL:=}"
: "${SEARCH_PROXY_TOKEN:=}"
: "${SEARCH_PROXY_TIMEOUT:=120}"
: "${SEARCH_PROXY_FETCH:=0}"
: "${SEARCH_PROXY_MAX_CHARS:=0}"
: "${SEARCH_PROXY_VERIFY_SSL:=true}"
: "${SERPER_API_KEY:=}"
: "${JINA_API_KEY:=}"
: "${SANDBOX_BASE_URL:=}"
: "${SANDBOX_API_TOKEN:=}"
: "${AIO_SANDBOX_BASE_URL:=}"
: "${LLM_BACKEND:=vllm}"
: "${VLLM_BASE_URL:=http://127.0.0.1:8004/v1}"
: "${VLLM_MODEL:=Qwen3.5-9B}"
: "${VLLM_API_KEY:=EMPTY}"
: "${VLLM_ENABLE_THINKING:=0}"
: "${VISION_BACKEND:=${LLM_BACKEND}}"
: "${VISION_BASE_URL:=${VLLM_BASE_URL}}"
: "${VISION_MODEL:=${VLLM_MODEL}}"
: "${VISION_API_KEY:=${VLLM_API_KEY}}"
: "${VISION_MODEL_CHECKPOINT:=${STUDENT_HF_CHECKPOINT}}"
: "${OPD_EXPERT_MODEL:=}"
: "${WIKI25_INDEX_PATH:=/root/sii-agent/data/wiki25/wiki25_fts.sqlite}"
: "${BROWSECOMP_INDEX_PATH:=/root/sii-agent/indexes/bm25}"
: "${SII_DISABLE_WIKI_TOOLS_WHEN_UNAVAILABLE:=1}"
: "${MEGATRON_PATH:=${REPO_ROOT}/third_party/Megatron-LM}"
: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_NUM_GPUS_PER_NODE:=4}"
if [[ "${START_VISION_SERVER}" =~ ^(1|true|True|yes|YES)$ ]]; then
  : "${ROLLOUT_NUM_GPUS:=1}"
else
  : "${ROLLOUT_NUM_GPUS:=2}"
fi
: "${COLOCATE:=0}"
if [[ "${COLOCATE}" =~ ^(1|true|True|yes|YES)$ ]]; then
  DEFAULT_RAY_NUM_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
else
  DEFAULT_RAY_NUM_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))
fi
: "${RAY_NUM_GPUS:=${DEFAULT_RAY_NUM_GPUS}}"
: "${NUM_GPUS_PER_NODE:=${RAY_NUM_GPUS}}"
: "${ROLLOUT_BATCH_SIZE:=32}"
: "${N_SAMPLES_PER_PROMPT:=8}"
: "${OVER_SAMPLING_BATCH_SIZE:=${ROLLOUT_BATCH_SIZE}}"
: "${ROLLOUT_MAX_PROMPT_LEN:=4096}"
: "${ROLLOUT_MAX_RESPONSE_LEN:=30000}"
: "${ROLLOUT_TEMPERATURE:=0.7}"
: "${GLOBAL_BATCH_SIZE:=8}"
: "${OVERRIDE_OPT_PARAM_SCHEDULER:=1}"
if [ -z "${NUM_ROLLOUT+x}" ]; then
  SAMPLES_PER_ROLLOUT=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
  if [ "${SAMPLES_PER_ROLLOUT}" -le 0 ] || [ "${GLOBAL_BATCH_SIZE}" -le 0 ]; then
    echo "Invalid rollout/train batch sizes: rollout_batch=${ROLLOUT_BATCH_SIZE}, n_samples=${N_SAMPLES_PER_PROMPT}, global_batch=${GLOBAL_BATCH_SIZE}" >&2
    exit 1
  fi
  ROLLOUTS_TO_RUN=$(((SII_SLIME_TRAIN_STEPS * GLOBAL_BATCH_SIZE + SAMPLES_PER_ROLLOUT - 1) / SAMPLES_PER_ROLLOUT))
  if [ "${ROLLOUTS_TO_RUN}" -lt 1 ]; then
    ROLLOUTS_TO_RUN=1
  fi
  START_ROLLOUT_GUESS=0
  if [ -f "${SLIME_SAVE}/latest_checkpointed_iteration.txt" ]; then
    LATEST_ROLLOUT="$(tr -cd '0-9' < "${SLIME_SAVE}/latest_checkpointed_iteration.txt")"
    if [ -n "${LATEST_ROLLOUT}" ]; then
      START_ROLLOUT_GUESS=$((LATEST_ROLLOUT + 1))
    fi
  fi
  NUM_ROLLOUT=$((START_ROLLOUT_GUESS + ROLLOUTS_TO_RUN))
fi
: "${TENSOR_MODEL_PARALLEL_SIZE:=4}"
: "${SEQUENCE_PARALLEL:=auto}"
: "${QKV_FORMAT:=bshd}"
: "${MICRO_BATCH_SIZE:=1}"
: "${USE_DYNAMIC_BATCH_SIZE:=0}"
: "${MASKED_SOFTMAX_FUSION:=0}"
: "${MAX_TOKENS_PER_GPU:=8192}"
: "${SGLANG_MEM_FRACTION_STATIC:=0.45}"
: "${TRANSFORMER_IMPL:=local}"
: "${EVAL_PROMPT_DATA:=/root/sii-agent/data/slime/sii_benchmark_answered_browser_eval.jsonl}"
: "${EVAL_PROMPT_NAME:=benchmark_answered}"
: "${EVAL_INTERVAL:=1}"
: "${EVAL_MAX_PROMPT_LEN:=4096}"
: "${EVAL_MAX_RESPONSE_LEN:=30000}"
: "${N_SAMPLES_PER_EVAL_PROMPT:=1}"
: "${SAVE_INTERVAL:=1}"

NVIDIA_SITE_LIB_ROOT="${NVIDIA_SITE_LIB_ROOT:-/root/myslime_env/lib/python3.12/site-packages/nvidia}"
if [ -d "${NVIDIA_SITE_LIB_ROOT}" ]; then
  NVIDIA_LIB_PATHS="$(find "${NVIDIA_SITE_LIB_ROOT}" -type d -path '*/lib' | paste -sd: -)"
  if [ -n "${NVIDIA_LIB_PATHS}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
fi

if [ ! -d "${SLIME_DIR}" ]; then
  echo "Missing slime checkout: ${SLIME_DIR}" >&2
  exit 1
fi
if [[ "${WIKI25_INDEX_PATH}" != /* && -e "${REPO_ROOT}/${WIKI25_INDEX_PATH}" ]]; then
  WIKI25_INDEX_PATH="${REPO_ROOT}/${WIKI25_INDEX_PATH}"
fi
if [[ "${BROWSECOMP_INDEX_PATH}" != /* && -e "${REPO_ROOT}/${BROWSECOMP_INDEX_PATH}" ]]; then
  BROWSECOMP_INDEX_PATH="${REPO_ROOT}/${BROWSECOMP_INDEX_PATH}"
fi
if [[ "${TEACHER_HF_CHECKPOINT}" == /* ]] && [ ! -d "${TEACHER_HF_CHECKPOINT}" ]; then
  echo "Missing teacher HF checkpoint: ${TEACHER_HF_CHECKPOINT}" >&2
  exit 1
fi
if [ ! -f "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" ]; then
  echo "Missing converted student checkpoint: ${STUDENT_TORCH_DIST}" >&2
  echo "Run: bash ${SCRIPT_DIR}/convert-qwen3.5-9B.sh" >&2
  exit 1
fi
if [ ! -f "${PROMPT_DATA}" ]; then
  echo "Missing prompt data: ${PROMPT_DATA}" >&2
  echo "Run: PYTHON_BIN=${SLIME_PYTHON} bash ${SCRIPT_DIR}/prepare-agent-opd-browser-data.sh" >&2
  exit 1
fi
if [ -n "${EVAL_PROMPT_DATA}" ] && [ ! -f "${EVAL_PROMPT_DATA}" ]; then
  echo "Missing eval prompt data: ${EVAL_PROMPT_DATA}" >&2
  echo "Run: PYTHON_BIN=${SLIME_PYTHON} bash ${SCRIPT_DIR}/prepare-agent-opd-browser-data.sh" >&2
  exit 1
fi

TEACHER_IP="127.0.0.1"
TEACHER_LOG="${SLIME_SAVE}/logs/teacher_sglang_${TEACHER_PORT}.log"
VISION_LOG="${SLIME_SAVE}/logs/vision_sglang_${VISION_PORT}.log"
mkdir -p "$(dirname "${TEACHER_LOG}")" "${SLIME_SAVE}"
TEACHER_PID=""
VISION_PID=""
RAY_STARTED_BY_SCRIPT=0

teacher_healthy() {
  curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null
}

vision_healthy() {
  curl -sf "${VISION_BASE_URL%/}/models" >/dev/null
}

validate_teacher_model() {
  if [[ "${VALIDATE_TEACHER_MODEL}" =~ ^(0|false|False|no|NO)$ ]]; then
    return 0
  fi
  TEACHER_IP="${TEACHER_IP}" TEACHER_PORT="${TEACHER_PORT}" TEACHER_HF_CHECKPOINT="${TEACHER_HF_CHECKPOINT}" "${SLIME_PYTHON}" - <<'PY'
import json
import os
import urllib.request

ip = os.environ["TEACHER_IP"]
port = os.environ["TEACHER_PORT"]
expected = os.path.realpath(os.environ["TEACHER_HF_CHECKPOINT"])
data = None
for path in ("get_model_info", "get_server_info"):
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/{path}", timeout=10) as resp:
            data = json.loads(resp.read())
        break
    except Exception:
        data = None
if not isinstance(data, dict):
    raise SystemExit(f"Could not read teacher model info from http://{ip}:{port}")
actual_raw = data.get("model_path") or data.get("tokenizer_path")
if not actual_raw:
    raise SystemExit(f"Teacher model info lacks model_path/tokenizer_path: {data}")
actual = os.path.realpath(str(actual_raw))
if actual != expected:
    raise SystemExit(f"Teacher model mismatch: expected {expected}, got {actual}")
print(f"Teacher model validated: {actual}")
PY
}

ray_healthy() {
  "${RAY_BIN}" status --address="${RAY_ADDRESS}" >/dev/null 2>&1
}

wait_for_ray_jobs_server() {
  local attempts="${RAY_JOB_SERVER_WAIT_ATTEMPTS:-60}"
  local delay="${RAY_JOB_SERVER_WAIT_SECONDS:-2}"
  for ((i = 1; i <= attempts; i++)); do
    if "${RAY_BIN}" job list --address="${RAY_JOB_ADDRESS}" >/dev/null 2>&1; then
      return 0
    fi
    echo "Waiting for Ray job server on ${RAY_JOB_ADDRESS} (${i}/${attempts})..."
    sleep "${delay}"
  done
  echo "Ray job server did not become ready on ${RAY_JOB_ADDRESS}" >&2
  return 1
}

cleanup() {
  set +e
  if [ -n "${TEACHER_PID}" ] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
    kill "${TEACHER_PID}"
    wait "${TEACHER_PID}" 2>/dev/null
  fi
  if [ -n "${VISION_PID}" ] && kill -0 "${VISION_PID}" 2>/dev/null; then
    kill "${VISION_PID}"
    wait "${VISION_PID}" 2>/dev/null
  fi
}
trap cleanup EXIT

cd "${SLIME_DIR}"
export PYTHONUNBUFFERED=1
export SEARCH_PROXY_URL SEARCH_PROXY_TOKEN SEARCH_PROXY_TIMEOUT SEARCH_PROXY_FETCH SEARCH_PROXY_MAX_CHARS SEARCH_PROXY_VERIFY_SSL
export SERPER_API_KEY JINA_API_KEY
export SANDBOX_BASE_URL SANDBOX_API_TOKEN AIO_SANDBOX_BASE_URL
export LLM_BACKEND VLLM_BASE_URL VLLM_MODEL VLLM_API_KEY VLLM_ENABLE_THINKING
export VISION_BACKEND VISION_BASE_URL VISION_MODEL VISION_API_KEY VISION_MODEL_CHECKPOINT OPD_EXPERT_MODEL
export WIKI25_INDEX_PATH BROWSECOMP_INDEX_PATH
export STUDENT_HF_CHECKPOINT TEACHER_PORT
export SII_SLIME_MAX_STEPS SII_SLIME_MAX_TURN_TOKENS SII_SLIME_MAX_OBSERVATION_CHARS SII_SLIME_MAX_REPEATS SII_SLIME_USE_TASK_REWARD
export SII_TEACHER_RM_CONCURRENCY SII_TEACHER_RM_RETRIES SII_TEACHER_RM_TOTAL_TIMEOUT SII_TEACHER_RM_CONNECT_TIMEOUT SII_TEACHER_RM_READ_TIMEOUT
export SII_DISABLE_VISION_TOOLS_WHEN_UNAVAILABLE SII_VISION_USE_ROLLOUT_ROUTER SII_DISABLE_WIKI_TOOLS_WHEN_UNAVAILABLE
export PYTHONPATH="${REPO_ROOT}:${SLIME_DIR}:${MEGATRON_PATH}:${PYTHONPATH:-}"

if teacher_healthy; then
  if [[ "${REUSE_TEACHER}" =~ ^(1|true|True|yes|YES)$ ]]; then
    echo "Reusing healthy teacher SGLang on ${TEACHER_IP}:${TEACHER_PORT}"
    validate_teacher_model
  else
    echo "Teacher port ${TEACHER_PORT} is already healthy. Set REUSE_TEACHER=1 to reuse it, or choose another TEACHER_PORT." >&2
    exit 1
  fi
else
  (
    export CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}"
    "${SLIME_PYTHON}" -m sglang.launch_server \
      --model-path "${TEACHER_HF_CHECKPOINT}" \
      --host 0.0.0.0 \
      --port "${TEACHER_PORT}" \
      --tp 2 \
      --chunked-prefill-size 4096 \
      --mem-fraction-static 0.72 \
      --trust-remote-code \
      > "${TEACHER_LOG}" 2>&1
  ) &
  TEACHER_PID=$!
fi

until teacher_healthy; do
  if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "Teacher server exited. Tail of ${TEACHER_LOG}:" >&2
    tail -n 80 "${TEACHER_LOG}" >&2 || true
    exit 1
  fi
  echo "Waiting for teacher SGLang on ${TEACHER_CUDA_VISIBLE_DEVICES}..."
  tail -n 10 "${TEACHER_LOG}" || true
  sleep 5
done
validate_teacher_model

"${SLIME_PYTHON}" - <<'PY'
import json
import os
import urllib.request
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ["STUDENT_HF_CHECKPOINT"], trust_remote_code=True)
ids = tok.encode("SII OPD teacher logprob smoke.", add_special_tokens=False)
payload = json.dumps({
    "input_ids": ids,
    "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
    "return_logprob": True,
    "logprob_start_len": 0,
}).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:{os.environ['TEACHER_PORT']}/generate",
    data=payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read())
logprobs = data["meta_info"]["input_token_logprobs"]
assert len(logprobs) >= len(ids), (len(logprobs), len(ids))
print(f"Teacher forced-logprob smoke ok: {len(logprobs)} input logprobs")
PY

if [[ "${START_VISION_SERVER}" =~ ^(1|true|True|yes|YES)$ ]]; then
  if vision_healthy; then
    echo "Reusing healthy vision SGLang at ${VISION_BASE_URL}"
  else
    (
      export CUDA_VISIBLE_DEVICES="${VISION_CUDA_VISIBLE_DEVICES}"
      "${SLIME_PYTHON}" -m sglang.launch_server \
        --model-path "${VISION_MODEL_CHECKPOINT}" \
        --host 0.0.0.0 \
        --port "${VISION_PORT}" \
        --tp "${VISION_TP}" \
        --context-length "${VISION_CONTEXT_LENGTH}" \
        --mem-fraction-static "${VISION_MEM_FRACTION_STATIC}" \
        --served-model-name "${VISION_MODEL}" \
        --trust-remote-code \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --mm-attention-backend sdpa \
        > "${VISION_LOG}" 2>&1
    ) &
    VISION_PID=$!
  fi
  until vision_healthy; do
    if [ -n "${VISION_PID}" ] && ! kill -0 "${VISION_PID}" 2>/dev/null; then
      echo "Vision server exited. Tail of ${VISION_LOG}:" >&2
      tail -n 120 "${VISION_LOG}" >&2 || true
      exit 1
    fi
    echo "Waiting for vision SGLang on ${VISION_CUDA_VISIBLE_DEVICES} at ${VISION_BASE_URL}..."
    tail -n 10 "${VISION_LOG}" || true
    sleep 5
  done
fi

export CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}"
if ray_healthy; then
  if [[ "${REUSE_RAY}" =~ ^(1|true|True|yes|YES)$ ]]; then
    echo "Reusing existing Ray cluster at ${RAY_ADDRESS}"
  else
    echo "Ray is already running at ${RAY_ADDRESS}. Set REUSE_RAY=1 to submit into it, or choose another RAY_GCS_PORT/RAY_DASHBOARD_PORT." >&2
    exit 1
  fi
else
  mkdir -p "${RAY_TEMP_DIR}"
  RAY_START_CMD=(
    "${RAY_BIN}" start
    --head
    --node-ip-address "${MASTER_ADDR}"
    --port "${RAY_GCS_PORT}"
    --num-gpus "${RAY_NUM_GPUS}"
    --disable-usage-stats
    --dashboard-host="${RAY_DASHBOARD_HOST}"
    --dashboard-port="${RAY_DASHBOARD_PORT}"
    --ray-client-server-port="${RAY_CLIENT_SERVER_PORT}"
    --min-worker-port="${RAY_MIN_WORKER_PORT}"
    --max-worker-port="${RAY_MAX_WORKER_PORT}"
    --temp-dir="${RAY_TEMP_DIR}"
  )
  if [ -n "${RAY_OBJECT_MANAGER_PORT}" ]; then
    RAY_START_CMD+=(--object-manager-port "${RAY_OBJECT_MANAGER_PORT}")
  fi
  if [ -n "${RAY_NODE_MANAGER_PORT}" ]; then
    RAY_START_CMD+=(--node-manager-port "${RAY_NODE_MANAGER_PORT}")
  fi
  if [ -n "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" ]; then
    RAY_START_CMD+=(--dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}")
  fi
  if [ -n "${RAY_DASHBOARD_AGENT_GRPC_PORT}" ]; then
    RAY_START_CMD+=(--dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}")
  fi
  if [ -n "${RAY_RUNTIME_ENV_AGENT_PORT}" ]; then
    RAY_START_CMD+=(--runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}")
  fi
  if [ -n "${RAY_METRICS_EXPORT_PORT}" ]; then
    RAY_START_CMD+=(--metrics-export-port "${RAY_METRICS_EXPORT_PORT}")
  fi
  "${RAY_START_CMD[@]}"
  RAY_STARTED_BY_SCRIPT=1
fi
wait_for_ray_jobs_server

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_HF_CHECKPOINT}"
   --ref-load "${STUDENT_TORCH_DIST}"
   --load "${SLIME_SAVE}"
   --save "${SLIME_SAVE}"
   --save-interval "${SAVE_INTERVAL}"
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key question
   --label-key answer
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
   --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --balance-data
   --custom-generate-function-path training.slime_sii_rollout.generate
   --save-debug-rollout-data "${SLIME_SAVE}/debug_rollout_{rollout_id}.pt"
)

RM_ARGS=(
   --custom-rm-path training.slime_sii_rollout.teacher_logprob_rm
   --custom-reward-post-process-path training.slime_sii_rollout.post_process_rewards
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

EVAL_ARGS=()
if [ -n "${EVAL_PROMPT_DATA}" ]; then
  EVAL_ARGS=(
     --eval-interval "${EVAL_INTERVAL:-1}"
     --eval-prompt-data "${EVAL_PROMPT_NAME}" "${EVAL_PROMPT_DATA}"
     --eval-input-key question
     --eval-label-key answer
     --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
     --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"
     --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  )
fi

SEQUENCE_PARALLEL_ARGS=()
if [[ "${SEQUENCE_PARALLEL}" == "auto" ]]; then
   if [ "${TENSOR_MODEL_PARALLEL_SIZE}" -gt 1 ]; then
      SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
   fi
elif [[ "${SEQUENCE_PARALLEL}" =~ ^(1|true|True|yes|YES)$ ]]; then
   SEQUENCE_PARALLEL_ARGS=(--sequence-parallel)
fi

PERF_ARGS=(
   --num-gpus-per-node "${NUM_GPUS_PER_NODE}"
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --qkv-format "${QKV_FORMAT}"
   "${SEQUENCE_PARALLEL_ARGS[@]}"
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --calculate-per-token-loss
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if [[ ! "${MASKED_SOFTMAX_FUSION}" =~ ^(1|true|True|yes|YES)$ ]]; then
   PERF_ARGS+=(--no-masked-softmax-fusion)
fi
if [[ "${USE_DYNAMIC_BATCH_SIZE}" =~ ^(1|true|True|yes|YES)$ ]]; then
   PERF_ARGS+=(--use-dynamic-batch-size)
else
   PERF_ARGS+=(--micro-batch-size "${MICRO_BATCH_SIZE}")
fi

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd
   --opd-type sglang
   --opd-kl-coef 1.0
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER}" =~ ^(1|true|True|yes|YES)$ ]]; then
   OPTIMIZER_ARGS+=(--override-opt-param-scheduler)
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

COLOCATE_ARGS=()
if [[ "${COLOCATE}" =~ ^(1|true|True|yes|YES)$ ]]; then
   COLOCATE_ARGS=(--colocate)
fi

MISC_ARGS=(
   --transformer-impl "${TRANSFORMER_IMPL}"
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
        "PYTHONPATH": "${REPO_ROOT}:${SLIME_DIR}:${MEGATRON_PATH}",
        "PATH": "${PATH}",
        "LD_LIBRARY_PATH": "${LD_LIBRARY_PATH:-}",
        "PYTORCH_CUDA_ALLOC_CONF": "${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "SEARCH_PROXY_URL": "${SEARCH_PROXY_URL}",
        "SEARCH_PROXY_TOKEN": "${SEARCH_PROXY_TOKEN}",
        "SEARCH_PROXY_TIMEOUT": "${SEARCH_PROXY_TIMEOUT}",
        "SEARCH_PROXY_FETCH": "${SEARCH_PROXY_FETCH}",
        "SEARCH_PROXY_MAX_CHARS": "${SEARCH_PROXY_MAX_CHARS}",
        "SEARCH_PROXY_VERIFY_SSL": "${SEARCH_PROXY_VERIFY_SSL}",
        "BROWSER_SERVICE_URL": "${BROWSER_SERVICE_URL:-}",
       "SERPER_API_KEY": "${SERPER_API_KEY}",
       "JINA_API_KEY": "${JINA_API_KEY}",
       "SANDBOX_BASE_URL": "${SANDBOX_BASE_URL}",
        "SANDBOX_API_TOKEN": "${SANDBOX_API_TOKEN}",
        "AIO_SANDBOX_BASE_URL": "${AIO_SANDBOX_BASE_URL}",
        "LLM_BACKEND": "${LLM_BACKEND}",
        "VLLM_BASE_URL": "${VLLM_BASE_URL}",
        "VLLM_MODEL": "${VLLM_MODEL}",
       "VLLM_API_KEY": "${VLLM_API_KEY}",
       "VLLM_ENABLE_THINKING": "${VLLM_ENABLE_THINKING}",
        "VISION_BACKEND": "${VISION_BACKEND}",
        "VISION_BASE_URL": "${VISION_BASE_URL}",
        "VISION_MODEL": "${VISION_MODEL}",
        "VISION_API_KEY": "${VISION_API_KEY}",
        "VISION_MODEL_CHECKPOINT": "${VISION_MODEL_CHECKPOINT}",
        "OPD_EXPERT_MODEL": "${OPD_EXPERT_MODEL}",
       "WIKI25_INDEX_PATH": "${WIKI25_INDEX_PATH}",
      "BROWSECOMP_INDEX_PATH": "${BROWSECOMP_INDEX_PATH}",
      "SII_SLIME_MAX_STEPS": "${SII_SLIME_MAX_STEPS}",
      "SII_SLIME_MAX_TURN_TOKENS": "${SII_SLIME_MAX_TURN_TOKENS}",
      "SII_SLIME_MAX_OBSERVATION_CHARS": "${SII_SLIME_MAX_OBSERVATION_CHARS}",
       "SII_SLIME_MAX_REPEATS": "${SII_SLIME_MAX_REPEATS}",
       "SII_SEARCH_SIMILARITY_THRESHOLD": "${SII_SEARCH_SIMILARITY_THRESHOLD}",
       "SII_SEARCH_MAX_SIMILAR_REPEATS": "${SII_SEARCH_MAX_SIMILAR_REPEATS}",
       "SII_SEARCH_MAX_CALLS_PER_TOOL": "${SII_SEARCH_MAX_CALLS_PER_TOOL}",
       "SII_SEARCH_MAX_CALLS_TOTAL": "${SII_SEARCH_MAX_CALLS_TOTAL}",
       "SII_SLIME_USE_TASK_REWARD": "${SII_SLIME_USE_TASK_REWARD}",
       "SII_TEACHER_RM_CONCURRENCY": "${SII_TEACHER_RM_CONCURRENCY}",
        "SII_TEACHER_RM_RETRIES": "${SII_TEACHER_RM_RETRIES}",
        "SII_TEACHER_RM_TOTAL_TIMEOUT": "${SII_TEACHER_RM_TOTAL_TIMEOUT}",
        "SII_TEACHER_RM_CONNECT_TIMEOUT": "${SII_TEACHER_RM_CONNECT_TIMEOUT}",
        "SII_TEACHER_RM_READ_TIMEOUT": "${SII_TEACHER_RM_READ_TIMEOUT}",
        "SII_TRAIN_CORRECT_ONLY": "${SII_TRAIN_CORRECT_ONLY}",
        "SII_TRAIN_DROP_TRUNCATED": "${SII_TRAIN_DROP_TRUNCATED}",
        "SII_TRAIN_MIXED_GROUPS_ONLY": "${SII_TRAIN_MIXED_GROUPS_ONLY}",
        "SII_TRAIN_REQUIRE_FINAL_ANSWER": "${SII_TRAIN_REQUIRE_FINAL_ANSWER}",
        "SII_TRAIN_REQUIRE_EVIDENCE_OPEN": "${SII_TRAIN_REQUIRE_EVIDENCE_OPEN}",
        "SII_TRAIN_MIN_SEARCH_CALLS_FOR_OPEN": "${SII_TRAIN_MIN_SEARCH_CALLS_FOR_OPEN}",
        "SII_TRAIN_MAX_TOTAL_TOKENS": "${SII_TRAIN_MAX_TOTAL_TOKENS}",
        "SII_TRAIN_MAX_RESPONSE_TOKENS": "${SII_TRAIN_MAX_RESPONSE_TOKENS}",
        "SII_TRAIN_MAX_TOOL_CALLS": "${SII_TRAIN_MAX_TOOL_CALLS}",
        "SII_DISABLE_VISION_TOOLS_WHEN_UNAVAILABLE": "${SII_DISABLE_VISION_TOOLS_WHEN_UNAVAILABLE}",
        "SII_VISION_USE_ROLLOUT_ROUTER": "${SII_VISION_USE_ROLLOUT_ROUTER}",
        "SII_DISABLE_WIKI_TOOLS_WHEN_UNAVAILABLE": "${SII_DISABLE_WIKI_TOOLS_WHEN_UNAVAILABLE}"
   }
}
EOF_JSON
)

echo "============================================================"
echo "SII-Agent slime OPD (agent rollout, browser-enabled retrieval)"
echo "============================================================"
echo "  agent harness  : ${REPO_ROOT}"
echo "  slime checkout : ${SLIME_DIR}"
echo "  slime python   : ${SLIME_PYTHON}"
echo "  ray bin        : ${RAY_BIN}"
echo "  student HF     : ${STUDENT_HF_CHECKPOINT}"
echo "  student torch  : ${STUDENT_TORCH_DIST}"
echo "  teacher HF     : ${TEACHER_HF_CHECKPOINT}"
echo "  teacher url    : http://${TEACHER_IP}:${TEACHER_PORT}/generate (reuse=${REUSE_TEACHER})"
echo "  ray cluster    : ${RAY_ADDRESS} jobs=${RAY_JOB_ADDRESS} temp=${RAY_TEMP_DIR} (reuse=${REUSE_RAY}, started_by_script=${RAY_STARTED_BY_SCRIPT})"
echo "  save dir       : ${SLIME_SAVE}"
echo "  train prompts  : ${PROMPT_DATA}"
echo "  eval prompts   : ${EVAL_PROMPT_NAME} ${EVAL_PROMPT_DATA:-<none>}"
echo "  student GPUs   : ${STUDENT_CUDA_VISIBLE_DEVICES} (actor=${ACTOR_NUM_GPUS_PER_NODE}, rollout=${ROLLOUT_NUM_GPUS}, ray=${RAY_NUM_GPUS}, num_gpus_per_node=${NUM_GPUS_PER_NODE}, colocate=${COLOCATE})"
echo "  teacher GPUs   : ${TEACHER_CUDA_VISIBLE_DEVICES}"
echo "  rollout        : num=${NUM_ROLLOUT} batch=${ROLLOUT_BATCH_SIZE} over_sample=${OVER_SAMPLING_BATCH_SIZE} n_samples=${N_SAMPLES_PER_PROMPT} T=${ROLLOUT_TEMPERATURE}"
echo "  limits         : train_steps=${SII_SLIME_TRAIN_STEPS} prompt=${ROLLOUT_MAX_PROMPT_LEN} response=${ROLLOUT_MAX_RESPONSE_LEN} agent_steps=${SII_SLIME_MAX_STEPS} obs_chars=${SII_SLIME_MAX_OBSERVATION_CHARS}"
echo "  transformer    : ${TRANSFORMER_IMPL} qkv=${QKV_FORMAT} dynamic_batch=${USE_DYNAMIC_BATCH_SIZE} micro_batch=${MICRO_BATCH_SIZE} masked_softmax_fusion=${MASKED_SOFTMAX_FUSION}"
echo "  OPD            : use_opd=1 type=sglang opd_kl_coef=1.0 task_reward=${SII_SLIME_USE_TASK_REWARD}"
echo "  train filter   : correct_only=${SII_TRAIN_CORRECT_ONLY} drop_truncated=${SII_TRAIN_DROP_TRUNCATED} mixed_groups=${SII_TRAIN_MIXED_GROUPS_ONLY} require_final=${SII_TRAIN_REQUIRE_FINAL_ANSWER} require_open=${SII_TRAIN_REQUIRE_EVIDENCE_OPEN} max_total=${SII_TRAIN_MAX_TOTAL_TOKENS} max_response=${SII_TRAIN_MAX_RESPONSE_TOKENS} max_tools=${SII_TRAIN_MAX_TOOL_CALLS}"
echo "  tools          : per-row metadata; browser_open/browser_open_many preserved, shell/memory excluded"
echo "  search proxy   : ${SEARCH_PROXY_URL:-<direct/none>} fetch=${SEARCH_PROXY_FETCH} max_chars=${SEARCH_PROXY_MAX_CHARS}"
echo "  vision endpoint: rollout_router=${SII_VISION_USE_ROLLOUT_ROUTER} fallback=${VISION_BACKEND} ${VISION_BASE_URL} ${VISION_MODEL} start_fallback=${START_VISION_SERVER} gpu=${VISION_CUDA_VISIBLE_DEVICES}"
echo "============================================================"

JOB_SUBMIT_CMD=(
   "${RAY_BIN}" job submit --address="${RAY_JOB_ADDRESS}"
   --runtime-env-json="${RUNTIME_ENV_JSON}"
   -- "${SLIME_PYTHON}" "${REPO_ROOT}/scripts/slime/run_train_qwen35.py"
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${COLOCATE_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${RM_ARGS[@]}"
)

JOB_SUBMIT_ATTEMPTS="${RAY_JOB_SUBMIT_ATTEMPTS:-5}"
for ((attempt = 1; attempt <= JOB_SUBMIT_ATTEMPTS; attempt++)); do
  if "${JOB_SUBMIT_CMD[@]}"; then
    exit 0
  fi
  if [ "${attempt}" = "${JOB_SUBMIT_ATTEMPTS}" ]; then
    echo "Ray job submit failed after ${JOB_SUBMIT_ATTEMPTS} attempts." >&2
    exit 1
  fi
  echo "Ray job submit failed; retrying (${attempt}/${JOB_SUBMIT_ATTEMPTS})..."
  sleep "$((attempt * 5))"
  wait_for_ray_jobs_server
done
