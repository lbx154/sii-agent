#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

: "${TRAIN_N:=512}"
: "${TRAIN_OFFSET:=0}"
: "${EVAL_N:=64}"
: "${EVAL_OFFSET:=0}"
: "${OUT_DIR:=${REPO_ROOT}/data/slime}"
: "${TRAIN_OUT:=${OUT_DIR}/sii_2wiki_train_${TRAIN_N}.jsonl}"
: "${BROWSECOMP_EVAL_OUT:=${OUT_DIR}/sii_browsecomp_test_${EVAL_N}.jsonl}"

mkdir -p "${OUT_DIR}"

python "${REPO_ROOT}/scripts/create_slime_sii_prompt_data.py" \
  --task 2wiki \
  --split train \
  --n "${TRAIN_N}" \
  --offset "${TRAIN_OFFSET}" \
  --out "${TRAIN_OUT}"

python "${REPO_ROOT}/scripts/create_slime_sii_prompt_data.py" \
  --task browsecomp-plus \
  --split test \
  --n "${EVAL_N}" \
  --offset "${EVAL_OFFSET}" \
  --out "${BROWSECOMP_EVAL_OUT}"

cat <<EOF
Prepared SII slime OPD prompt data:
  train: ${TRAIN_OUT}
  eval:  ${BROWSECOMP_EVAL_OUT}

Example training launch:
  PROMPT_DATA=${TRAIN_OUT} \\
  EVAL_PROMPT_DATA=${BROWSECOMP_EVAL_OUT} \\
  bash ${SCRIPT_DIR}/run-sii-qwen3.5-9B-opd.sh
EOF
