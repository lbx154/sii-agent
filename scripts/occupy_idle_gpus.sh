#!/usr/bin/env bash
# 占卡脚本: start the idle-GPU holder with a clear argv[0].
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec -a "占卡脚本" python "${SCRIPT_DIR}/occupy_idle_gpus.py" "$@"
