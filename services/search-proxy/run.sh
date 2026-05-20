#!/usr/bin/env bash
# 在 CPU 服务器（有外网）上启动 search-proxy。
#
# 默认监听 127.0.0.1:8090 —— 因为我们要走 SSH/VSCode 端口转发，
# 没有任何理由把它暴露在公网。
#
# 使用：
#   export SERPER_API_KEY=xxx
#   export JINA_API_KEY=xxx           # 可选
#   export PROXY_API_TOKEN=xxx        # 可选；建议设置一个随机字符串
#   ./run.sh

set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[setup] creating .venv ..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"

echo "[run] uvicorn app.main:app --host $HOST --port $PORT"
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --no-access-log
