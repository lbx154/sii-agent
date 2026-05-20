#!/bin/bash
# Pure GRPO RL launcher for SII-Agent + Qwen3.5-9B.
#
# What this does:
#   * Spins up slime + Ray with a *single* SGLang rollout engine (no teacher).
#   * Uses `training.slime_sii_rl.generate` as the rollout function so every
#     rollout exercises the SII-Agent tool registry exactly like benchmark
#     evaluation does (web_search / wiki_search / browser_open* / final_answer).
#   * Optimises with GRPO + KL-to-ref using `shaped_reward_rm`. No OPD, no
#     teacher logprob distillation, no remote RM server.
#
# Recipe rationale (see plan.md):
#   * No SFT warmup — the harness already gets >=40% on these tasks; the goal
#     of RL here is just to nudge tool-calling and search patterns. Fast and
#     low-risk; any positive delta is a win.
#   * Train set defaults to the merged browsecomp-plus + mmsearch train jsonl
#     (5686 prompts) because it most closely mirrors the target benchmark
#     (long multi-hop research questions + visual factual QA).
#   * Reward is dominated by task correctness (1.0) with light shaping to
#     penalise redundant searches and reward clean final_answer calls —
#     directly targeting the benchmark's observed failure modes.
#   * n_samples_per_prompt=8 so GRPO has within-group variance even on hard
#     prompts where mean reward may be very small.
#
# Quick smoke (32 prompts, ~30 min to first checkpoint):
#   PROMPT_DATA=${REPO_ROOT}/data/slime/sii_browsecomp_mmsearch_train_smoke32.jsonl \
#   NUM_ROLLOUT=4 ROLLOUT_BATCH_SIZE=4 N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 \
#   bash scripts/slime/run-sii-qwen35-9b-rl.sh
#
# Required environment before running:
#   1. SGLang + slime + Megatron deps installed in /root/myslime_env.
#   2. Tool services up: browser-service / search-proxy (or Serper key).
#      Pass through the same env vars you use for `python -m evaluation.run_eval`.
#   3. Torch-dist student checkpoint converted via
#         bash scripts/slime/convert-qwen3.5-9B.sh
#   4. Prompt data jsonl (one line per prompt with keys question, answer,
#      metadata). Build with `scripts/create_slime_sii_prompt_data.py`.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
SLIME_DIR="${REPO_ROOT}/third_party/slime"
source "${SCRIPT_DIR}/qwen3.5-9B.sh"

# ---------------------------------------------------------------------------
# Paths and resources
# ---------------------------------------------------------------------------
: "${STUDENT_HF_CHECKPOINT:=${REPO_ROOT}/Qwen3.5-9B}"
: "${STUDENT_TORCH_DIST:=${REPO_ROOT}/Qwen3.5-9B_torch_dist}"
: "${SLIME_SAVE:=${REPO_ROOT}/saves/qwen35-9b/rl_grpo}"
# Train data: browsecomp-plus + mmsearch joint (5686 prompts).
# For a fast pipeline smoke test, override with the smoke32 file (32 prompts).
#   PROMPT_DATA=${REPO_ROOT}/data/slime/sii_browsecomp_mmsearch_train_smoke32.jsonl
: "${PROMPT_DATA:=${REPO_ROOT}/data/slime/sii_browsecomp_mmsearch_train_all.jsonl}"
# Eval set defaults to BrowseComp-Plus test slice (64 prompts).
: "${EVAL_PROMPT_DATA:=${REPO_ROOT}/data/slime/sii_browsecomp_test_64.jsonl}"
: "${MEGATRON_PATH:=${REPO_ROOT}/third_party/Megatron-LM}"

# Avoid GPU 0 (used by external SGLang for eval) and GPU 6/7 (used by others).
: "${STUDENT_CUDA_VISIBLE_DEVICES:=1,2,3,4}"
: "${MASTER_ADDR:=127.0.0.1}"

# ---------------------------------------------------------------------------
# Rollout / training hyperparameters
# ---------------------------------------------------------------------------
# Resources
: "${ACTOR_NUM_NODES:=1}"
: "${ACTOR_NUM_GPUS_PER_NODE:=2}"
: "${ROLLOUT_NUM_GPUS:=2}"
: "${RAY_NUM_GPUS:=$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
: "${TENSOR_MODEL_PARALLEL_SIZE:=2}"
: "${MAX_TOKENS_PER_GPU:=8192}"
: "${SGLANG_MEM_FRACTION_STATIC:=0.7}"

# Rollout shape — defaults aimed at ~16 concurrent rollouts on the rollout engine
#   (ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT = concurrent rollouts per iter)
: "${NUM_ROLLOUT:=64}"
: "${ROLLOUT_BATCH_SIZE:=2}"
: "${N_SAMPLES_PER_PROMPT:=8}"          # GRPO group size, also concurrent multiplier
: "${ROLLOUT_MAX_PROMPT_LEN:=4096}"
: "${ROLLOUT_MAX_RESPONSE_LEN:=24000}"
: "${ROLLOUT_TEMPERATURE:=1.0}"         # high temp for exploration
: "${GLOBAL_BATCH_SIZE:=16}"

# Agent rollout limits (read by training.slime_sii_rl.generate)
: "${SII_RL_MAX_STEPS:=12}"
: "${SII_RL_MAX_TURN_TOKENS:=2048}"
: "${SII_RL_MAX_OBSERVATION_CHARS:=3000}"
: "${SII_RL_MAX_REPEATS:=2}"

# Reward shaping (read by training.slime_sii_rl.shaped_reward_rm)
: "${SII_RL_REWARD_TASK:=1.0}"
: "${SII_RL_REWARD_FINAL:=0.1}"
: "${SII_RL_REWARD_FINAL_TEXT:=0.02}"
: "${SII_RL_REWARD_TOOL_USE:=0.05}"
: "${SII_RL_REWARD_NO_TOOL:=0.05}"
: "${SII_RL_REWARD_FORMAT:=0.3}"
: "${SII_RL_REWARD_REDUNDANT:=0.1}"
: "${SII_RL_REWARD_STEP:=0.02}"
: "${SII_RL_REWARD_STEP_BUDGET:=5}"
: "${SII_RL_REWARD_CLIP_LOW:=-0.5}"
: "${SII_RL_REWARD_CLIP_HIGH:=1.5}"

# Tool service config (forwarded to Ray workers).
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
: "${WIKI25_INDEX_PATH:=${REPO_ROOT}/data/wiki25/wiki25_fts.sqlite}"
: "${BROWSECOMP_INDEX_PATH:=${REPO_ROOT}/indexes/bm25}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
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
  echo "Available datasets under data/slime/:" >&2
  ls "${REPO_ROOT}/data/slime/" 2>/dev/null | sed 's/^/  /' >&2
  echo "Build with: python ${REPO_ROOT}/scripts/create_slime_sii_prompt_data.py \\" >&2
  echo "             --task browsecomp-plus --split train --out ${PROMPT_DATA}" >&2
  exit 1
fi
if [ -n "${EVAL_PROMPT_DATA}" ] && [ ! -f "${EVAL_PROMPT_DATA}" ]; then
  echo "Missing eval prompt data: ${EVAL_PROMPT_DATA}" >&2
  exit 1
fi

mkdir -p "${SLIME_SAVE}/logs"

cleanup() {
  set +e
  # Only tear down Ray if launcher is exiting with an error AND user requested
  # an automatic cleanup. With --no-wait, normal EXIT just means "submission
  # done, job is still running on the cluster" — we must NOT kill Ray here.
  local rc=$?
  if [ "${SII_RL_KEEP_RAY:-1}" = "0" ] && [ "$rc" -ne 0 ]; then
    CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}" ray stop --force >/dev/null 2>&1
  fi
}
trap cleanup EXIT

cd "${SLIME_DIR}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO_ROOT}:${SLIME_DIR}:${MEGATRON_PATH}:${PYTHONPATH:-}"
export STUDENT_HF_CHECKPOINT
export SII_RL_MAX_STEPS SII_RL_MAX_TURN_TOKENS SII_RL_MAX_OBSERVATION_CHARS SII_RL_MAX_REPEATS
export SII_RL_REWARD_TASK SII_RL_REWARD_FINAL SII_RL_REWARD_FINAL_TEXT
export SII_RL_REWARD_TOOL_USE SII_RL_REWARD_NO_TOOL SII_RL_REWARD_FORMAT
export SII_RL_REWARD_REDUNDANT SII_RL_REWARD_STEP SII_RL_REWARD_STEP_BUDGET
export SII_RL_REWARD_CLIP_LOW SII_RL_REWARD_CLIP_HIGH
export SEARCH_PROXY_URL SEARCH_PROXY_TOKEN SEARCH_PROXY_TIMEOUT SEARCH_PROXY_FETCH \
       SEARCH_PROXY_MAX_CHARS SEARCH_PROXY_VERIFY_SSL
export SERPER_API_KEY JINA_API_KEY
export SANDBOX_BASE_URL SANDBOX_API_TOKEN AIO_SANDBOX_BASE_URL
export WIKI25_INDEX_PATH BROWSECOMP_INDEX_PATH

# ---------------------------------------------------------------------------
# Start Ray (single head, all GPUs assigned to the student)
# ---------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${STUDENT_CUDA_VISIBLE_DEVICES}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

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
  --apply-chat-template
  --rollout-shuffle
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature "${ROLLOUT_TEMPERATURE}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
  --custom-generate-function-path training.slime_sii_rl.generate
  --save-debug-rollout-data "${SLIME_SAVE}/debug_rollout_{rollout_id}.pt"
)

# Pure rule-based reward (no teacher, no remote RM).
RM_ARGS=(
  --custom-rm-path training.slime_sii_rl.shaped_reward_rm
)

EVAL_ARGS=()
if [ -n "${EVAL_PROMPT_DATA}" ]; then
  EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-4}"
    --eval-prompt-data benchmark "${EVAL_PROMPT_DATA}"
    --eval-input-key question
    --eval-label-key answer
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
    --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN:-4096}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-12000}"
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

# Pure GRPO — NO KL-to-ref, NO ref model. NOTE: --use-opd is intentionally absent.
# Both kl_coef and kl_loss_coef are 0; advantages are pure group-normalized rewards.
# Slime's compute_advantages_and_returns has been patched locally to handle the
# all-None log-probs case (see third_party/slime/slime/backends/megatron_utils/loss.py).
GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-coef 0
  --kl-loss-coef 0
  --entropy-coef "${ENTROPY_COEF:-0.0}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR:-1e-6}"
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
    "PYTHONPATH": "${REPO_ROOT}:${SLIME_DIR}:${MEGATRON_PATH}",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "SEARCH_PROXY_URL": "${SEARCH_PROXY_URL}",
    "SEARCH_PROXY_TOKEN": "${SEARCH_PROXY_TOKEN}",
    "SEARCH_PROXY_TIMEOUT": "${SEARCH_PROXY_TIMEOUT}",
    "SEARCH_PROXY_FETCH": "${SEARCH_PROXY_FETCH}",
    "SEARCH_PROXY_MAX_CHARS": "${SEARCH_PROXY_MAX_CHARS}",
    "SEARCH_PROXY_VERIFY_SSL": "${SEARCH_PROXY_VERIFY_SSL}",
    "SERPER_API_KEY": "${SERPER_API_KEY}",
    "JINA_API_KEY": "${JINA_API_KEY}",
    "SANDBOX_BASE_URL": "${SANDBOX_BASE_URL}",
    "SANDBOX_API_TOKEN": "${SANDBOX_API_TOKEN}",
    "AIO_SANDBOX_BASE_URL": "${AIO_SANDBOX_BASE_URL}",
    "WIKI25_INDEX_PATH": "${WIKI25_INDEX_PATH}",
    "BROWSECOMP_INDEX_PATH": "${BROWSECOMP_INDEX_PATH}",
    "SII_RL_MAX_STEPS": "${SII_RL_MAX_STEPS}",
    "SII_RL_MAX_TURN_TOKENS": "${SII_RL_MAX_TURN_TOKENS}",
    "SII_RL_MAX_OBSERVATION_CHARS": "${SII_RL_MAX_OBSERVATION_CHARS}",
    "SII_RL_MAX_REPEATS": "${SII_RL_MAX_REPEATS}",
    "SII_RL_REWARD_TASK": "${SII_RL_REWARD_TASK}",
    "SII_RL_REWARD_FINAL": "${SII_RL_REWARD_FINAL}",
    "SII_RL_REWARD_FINAL_TEXT": "${SII_RL_REWARD_FINAL_TEXT}",
    "SII_RL_REWARD_TOOL_USE": "${SII_RL_REWARD_TOOL_USE}",
    "SII_RL_REWARD_NO_TOOL": "${SII_RL_REWARD_NO_TOOL}",
    "SII_RL_REWARD_FORMAT": "${SII_RL_REWARD_FORMAT}",
    "SII_RL_REWARD_REDUNDANT": "${SII_RL_REWARD_REDUNDANT}",
    "SII_RL_REWARD_STEP": "${SII_RL_REWARD_STEP}",
    "SII_RL_REWARD_STEP_BUDGET": "${SII_RL_REWARD_STEP_BUDGET}",
    "SII_RL_REWARD_CLIP_LOW": "${SII_RL_REWARD_CLIP_LOW}",
    "SII_RL_REWARD_CLIP_HIGH": "${SII_RL_REWARD_CLIP_HIGH}"
  }
}
EOF_JSON
)

echo "============================================================"
echo "SII-Agent Pure GRPO RL"
echo "============================================================"
echo "  student HF  : ${STUDENT_HF_CHECKPOINT}"
echo "  ref/torch   : ${STUDENT_TORCH_DIST}"
echo "  save        : ${SLIME_SAVE}"
echo "  prompt data : ${PROMPT_DATA}"
echo "  eval data   : ${EVAL_PROMPT_DATA:-<none>}"
echo "  GPUs (student): ${STUDENT_CUDA_VISIBLE_DEVICES}"
echo "  GRPO group  : n_samples=${N_SAMPLES_PER_PROMPT} batch=${ROLLOUT_BATCH_SIZE} global=${GLOBAL_BATCH_SIZE}"
echo "  rollout     : T=${ROLLOUT_TEMPERATURE} max_resp=${ROLLOUT_MAX_RESPONSE_LEN} max_steps=${SII_RL_MAX_STEPS}"
echo "  reward      : task=${SII_RL_REWARD_TASK} final=${SII_RL_REWARD_FINAL} tool=${SII_RL_REWARD_TOOL_USE} redundant=-${SII_RL_REWARD_REDUNDANT}/call"
echo "  kl_loss_coef: ${KL_LOSS_COEF:-0.001}"
echo "============================================================"

ray job submit --address="http://127.0.0.1:8265" \
  --no-wait \
  --submission-id="${RAY_JOB_SUBMISSION_ID:-sii-rl-$(date +%s)}" \
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
