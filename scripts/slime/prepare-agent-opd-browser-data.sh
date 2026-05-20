#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"

: "${OUT_DIR:=${REPO_ROOT}/data/slime}"
: "${BROWSECOMP_OUT:=${OUT_DIR}/sii_browsecomp_plus_all.jsonl}"
: "${MMSEARCH_OUT:=${OUT_DIR}/sii_mmsearch_train_all_browser.jsonl}"
: "${TRAIN_OUT:=${OUT_DIR}/sii_agent_browser_opd_train.jsonl}"
: "${BENCHMARK_CSV:=${REPO_ROOT}/data/benchmark_answered.csv}"
: "${EVAL_OUT:=${OUT_DIR}/sii_benchmark_answered_browser_eval.jsonl}"
: "${PYTHON_BIN:=python}"

BROWSECOMP_TOOLS="search,final_answer"
MMSEARCH_TOOLS="visual_web_search,image_to_text,image_to_search_queries,reverse_image_search,web_search,wiki_search,wiki_page,browser_open,browser_open_many,final_answer"

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/create_slime_sii_prompt_data.py" \
  --task browsecomp-plus \
  --split test \
  --n 0 \
  --out "${BROWSECOMP_OUT}" \
  --allowed-tools "${BROWSECOMP_TOOLS}" \
  --chat-prompt

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/create_slime_sii_prompt_data.py" \
  --task mmsearch \
  --split train \
  --n 0 \
  --out "${MMSEARCH_OUT}" \
  --allowed-tools "${MMSEARCH_TOOLS}" \
  --chat-prompt

"${PYTHON_BIN}" - "${BROWSECOMP_OUT}" "${MMSEARCH_OUT}" "${TRAIN_OUT}" <<'PY'
from pathlib import Path
import sys

sources = [Path(sys.argv[1]), Path(sys.argv[2])]
out = Path(sys.argv[3])
with out.open("w", encoding="utf-8") as dst:
    for source in sources:
        with source.open("r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    dst.write(line if line.endswith("\n") else line + "\n")
PY

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/create_slime_benchmark_prompt_data.py" \
  --csv "${BENCHMARK_CSV}" \
  --out "${EVAL_OUT}" \
  --chat-prompt

"${PYTHON_BIN}" - "${TRAIN_OUT}" "${EVAL_OUT}" <<'PY'
from collections import Counter
from pathlib import Path
import json
import sys

for raw in sys.argv[1:]:
    path = Path(raw)
    counts = Counter()
    tool_sets = Counter()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            meta = item.get("metadata") or {}
            counts[meta.get("task", "unknown")] += 1
            tool_sets[tuple(meta.get("allowed_tools") or [])] += 1
    print(path)
    print("  rows:", sum(counts.values()))
    print("  tasks:", dict(counts))
    for tools, count in tool_sets.most_common():
        print(f"  tools[{count}]: {','.join(tools)}")
PY

cat <<EOF
Prepared SII slime agent OPD data with browser-enabled retrieval:
  train: ${TRAIN_OUT}
  eval:  ${EVAL_OUT}

Example launch:
  PROMPT_DATA=${TRAIN_OUT} \\
  EVAL_PROMPT_DATA=${EVAL_OUT} \\
  EVAL_PROMPT_NAME=benchmark_answered \\
  REUSE_TEACHER=1 \\
  SLIME_PYTHON=/root/myslime_env/bin/python \\
  bash ${SCRIPT_DIR}/run-sii-qwen3.5-9B-opd.sh
EOF
