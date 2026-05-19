#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_ROOT}/third_party/slime"
source "${SCRIPT_DIR}/qwen3.5-9B.sh"

: "${STUDENT_HF_CHECKPOINT:=/root/sii-agent/Qwen3.5-9B}"
: "${TEACHER_HF_CHECKPOINT:=/root/sii-agent/Qwen3.5-27B}"
: "${STUDENT_TORCH_DIST:=/root/sii-agent/Qwen3.5-9B_torch_dist}"
: "${SLIME_SAVE:=/root/sii-agent/Qwen3.5-9B_sii_slime_opd}"
: "${PROMPT_DATA:=/root/sii-agent/data/slime/sii_2wiki_train_smoke.jsonl}"
: "${TEACHER_CUDA_VISIBLE_DEVICES:=6,7}"
: "${STUDENT_CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${TEACHER_PORT:=13141}"
: "${TEACHER_TP:=2}"
: "${TEACHER_MEM_FRACTION_STATIC:=0.72}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${SII_SLIME_MAX_STEPS:=5}"
: "${SII_SLIME_MAX_TURN_TOKENS:=1024}"
: "${SII_SLIME_MAX_OBSERVATION_CHARS:=4000}"
: "${SII_SLIME_USE_TASK_REWARD:=1}"
: "${SII_SLIME_FORCE_FINAL_STEP:=1}"
: "${SEARCH_BACKENDS:=wiki}"
: "${WIKI25_INDEX_PATH:=/root/sii-agent/data/wiki25/wiki25_fts.sqlite}"
: "${BROWSECOMP_INDEX_PATH:=/root/sii-agent/data/browsecomp-plus/browsecomp_fts.sqlite}"
: "${MEGATRON_PATH:=${REPO_ROOT}/third_party/Megatron-LM}"
: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_NUM_GPUS_PER_NODE:=2}"
: "${ROLLOUT_NUM_GPUS:=2}"
: "${RAY_NUM_GPUS:=$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
: "${NUM_ROLLOUT:=2}"
: "${ROLLOUT_BATCH_SIZE:=2}"
: "${N_SAMPLES_PER_PROMPT:=2}"
: "${ROLLOUT_MAX_PROMPT_LEN:=4096}"
: "${ROLLOUT_MAX_RESPONSE_LEN:=8192}"
: "${ROLLOUT_TEMPERATURE:=0.7}"
: "${GLOBAL_BATCH_SIZE:=4}"
: "${TENSOR_MODEL_PARALLEL_SIZE:=2}"
: "${MAX_TOKENS_PER_GPU:=8192}"
: "${SGLANG_MEM_FRACTION_STATIC:=0.45}"
: "${SAVE_INTERVAL:=1}"
: "${EVAL_PROMPT_DATA:=}"
: "${EVAL_INTERVAL:=}"
: "${EVAL_MAX_PROMPT_LEN:=4096}"
: "${EVAL_MAX_RESPONSE_LEN:=8192}"
: "${N_SAMPLES_PER_EVAL_PROMPT:=1}"

if [ ! -d "${SLIME_DIR}" ]; then
  echo "Missing slime checkout: ${SLIME_DIR}" >&2
  exit 1
fi
if [ ! -f "${STUDENT_TORCH_DIST}/latest_checkpointed_iteration.txt" ]; then
  echo "Missing converted student checkpoint: ${STUDENT_TORCH_DIST}" >&2
  echo "Run: bash ${SCRIPT_DIR}/convert-qwen3.5-9B.sh" >&2
  exit 1
fi
if [ ! -f "${PROMPT_DATA}" ]; then
  echo "Missing prompt data: ${PROMPT_DATA}" >&2
  echo "Run: python ${REPO_ROOT}/scripts/create_slime_sii_prompt_data.py --task 2wiki --split train --n 32 --out ${PROMPT_DATA}" >&2
  exit 1
fi
if [ -n "${EVAL_PROMPT_DATA}" ] && [ ! -f "${EVAL_PROMPT_DATA}" ]; then
  echo "Missing eval prompt data: ${EVAL_PROMPT_DATA}" >&2
  echo "Run: python ${REPO_ROOT}/scripts/create_slime_sii_prompt_data.py --task browsecomp-plus --split test --n 64 --out ${EVAL_PROMPT_DATA}" >&2
  exit 1
fi

TEACHER_IP="127.0.0.1"
TEACHER_LOG="${SLIME_SAVE}/logs/teacher_sglang_${TEACHER_PORT}.log"
mkdir -p "$(dirname "${TEACHER_LOG}")" "${SLIME_SAVE}"
TEACHER_PID=""

cleanup() {
  set +e
  if [ -n "${TEACHER_PID}" ] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
    kill "${TEACHER_PID}"
    wait "${TEACHER_PID}" 2>/dev/null
  fi
  CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" ray stop --force >/dev/null 2>&1
}
trap cleanup EXIT

cd "${SLIME_DIR}"
export PYTHONBUFFERED=16
export SEARCH_BACKENDS WIKI25_INDEX_PATH BROWSECOMP_INDEX_PATH
export STUDENT_HF_CHECKPOINT TEACHER_PORT
export SII_SLIME_MAX_STEPS SII_SLIME_MAX_TURN_TOKENS SII_SLIME_MAX_OBSERVATION_CHARS SII_SLIME_USE_TASK_REWARD SII_SLIME_FORCE_FINAL_STEP
export PYTHONPATH="${REPO_ROOT}:${MEGATRON_PATH}:${SLIME_DIR}:${PYTHONPATH:-}"

(
  export CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}"
  python -m sglang.launch_server \
    --model-path "${TEACHER_HF_CHECKPOINT}" \
    --host 0.0.0.0 \
    --port "${TEACHER_PORT}" \
    --tp "${TEACHER_TP}" \
    --chunked-prefill-size 4096 \
    --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
    --trust-remote-code \
    > "${TEACHER_LOG}" 2>&1
) &
TEACHER_PID=$!

until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
  if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
    echo "Teacher server exited. Tail of ${TEACHER_LOG}:" >&2
    tail -n 80 "${TEACHER_LOG}" >&2 || true
    exit 1
  fi
  echo "Waiting for teacher SGLang on ${TEACHER_CUDA_VISIBLE_DEVICES}..."
  tail -n 10 "${TEACHER_LOG}" || true
  sleep 5
done

python - <<'PY'
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

export CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
   --reward-key scalar_reward
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

EVAL_ARGS=()
if [ -n "${EVAL_PROMPT_DATA}" ]; then
  EVAL_ARGS=(
     --eval-interval "${EVAL_INTERVAL:-1}"
     --eval-prompt-data browsecomp-plus "${EVAL_PROMPT_DATA}"
     --eval-input-key question
     --eval-label-key answer
     --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
     --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"
     --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
  )
fi

PERF_ARGS=(
   --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
      "PYTHONPATH": "${REPO_ROOT}:${MEGATRON_PATH}:${SLIME_DIR}",
     "CUDA_DEVICE_MAX_CONNECTIONS": "1",
     "SEARCH_BACKENDS": "${SEARCH_BACKENDS}",
     "WIKI25_INDEX_PATH": "${WIKI25_INDEX_PATH}",
     "BROWSECOMP_INDEX_PATH": "${BROWSECOMP_INDEX_PATH}",
     "SII_SLIME_MAX_STEPS": "${SII_SLIME_MAX_STEPS}",
      "SII_SLIME_MAX_TURN_TOKENS": "${SII_SLIME_MAX_TURN_TOKENS}",
      "SII_SLIME_MAX_OBSERVATION_CHARS": "${SII_SLIME_MAX_OBSERVATION_CHARS}",
     "SII_SLIME_USE_TASK_REWARD": "${SII_SLIME_USE_TASK_REWARD}",
     "SII_SLIME_FORCE_FINAL_STEP": "${SII_SLIME_FORCE_FINAL_STEP}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python "${REPO_ROOT}/scripts/slime/run_train_qwen35.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"
