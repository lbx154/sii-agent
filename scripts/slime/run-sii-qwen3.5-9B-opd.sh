#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_ROOT}/third_party/slime"
source "${SCRIPT_DIR}/qwen3.5-9B.sh"

: "${STUDENT_HF_CHECKPOINT:=/root/models/Qwen3.5-9B}"
: "${TEACHER_HF_CHECKPOINT:=/root/models/Qwen3.5-27B}"
: "${STUDENT_TORCH_DIST:=/root/models/Qwen3.5-9B_torch_dist}"
: "${SLIME_SAVE:=/root/models/Qwen3.5-9B_sii_slime_opd}"
: "${PROMPT_DATA:=/root/sii-agent/data/slime/sii_2wiki_train_smoke.jsonl}"
: "${TEACHER_CUDA_VISIBLE_DEVICES:=6,7}"
: "${STUDENT_CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${TEACHER_PORT:=13141}"
: "${MASTER_ADDR:=127.0.0.1}"
: "${SII_SLIME_MAX_STEPS:=5}"
: "${SII_SLIME_MAX_TURN_TOKENS:=1024}"
: "${SII_SLIME_MAX_OBSERVATION_CHARS:=4000}"
: "${SII_SLIME_USE_TASK_REWARD:=1}"
: "${SEARCH_BACKENDS:=wiki}"
: "${MEGATRON_PATH:=/root/Megatron-LM}"

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
export SEARCH_BACKENDS
export STUDENT_HF_CHECKPOINT TEACHER_PORT
export SII_SLIME_MAX_STEPS SII_SLIME_MAX_TURN_TOKENS SII_SLIME_MAX_OBSERVATION_CHARS SII_SLIME_USE_TASK_REWARD
export PYTHONPATH="${MEGATRON_PATH}:${REPO_ROOT}:${SLIME_DIR}:${PYTHONPATH:-}"

(
  export CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}"
  python -m sglang.launch_server \
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
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

CKPT_ARGS=(
   --hf-checkpoint "${STUDENT_HF_CHECKPOINT}"
   --ref-load "${STUDENT_TORCH_DIST}"
   --load "${SLIME_SAVE}"
   --save "${SLIME_SAVE}"
   --save-interval 1
)

ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key question
   --label-key answer
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout 2
   --rollout-batch-size 2
   --n-samples-per-prompt 2
   --rollout-max-prompt-len 4096
   --rollout-max-response-len 8192
   --rollout-temperature 0.7
   --global-batch-size 4
   --balance-data
   --custom-generate-function-path training.slime_sii_rollout.generate
   --save-debug-rollout-data "${SLIME_SAVE}/debug_rollout_{rollout_id}.pt"
)

RM_ARGS=(
   --custom-rm-path training.slime_sii_rollout.teacher_logprob_rm
   --custom-reward-post-process-path training.slime_sii_rollout.post_process_rewards
   --rm-url "http://${TEACHER_IP}:${TEACHER_PORT}/generate"
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
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
   --max-tokens-per-gpu 8192
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
   --sglang-mem-fraction-static 0.45
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
    "PYTHONPATH": "${MEGATRON_PATH}:${REPO_ROOT}:${SLIME_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SEARCH_BACKENDS": "${SEARCH_BACKENDS}",
    "SII_SLIME_MAX_STEPS": "${SII_SLIME_MAX_STEPS}",
    "SII_SLIME_MAX_TURN_TOKENS": "${SII_SLIME_MAX_TURN_TOKENS}",
    "SII_SLIME_MAX_OBSERVATION_CHARS": "${SII_SLIME_MAX_OBSERVATION_CHARS}",
    "SII_SLIME_USE_TASK_REWARD": "${SII_SLIME_USE_TASK_REWARD}"
  }
}
EOF_JSON
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 2 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"
