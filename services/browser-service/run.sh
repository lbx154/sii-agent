#!/usr/bin/env bash
# 一键启动脚本（macOS / Linux / 容器内 通用）
#
# 行为：
#   - 自动识别是否在容器内：
#       * 普通主机 → 创建 .venv 虚拟环境
#       * 容器内    → 直接用系统 Python（venv 在受限容器里反而麻烦）
#   - 自动尝试安装 Playwright Chromium 二进制
#   - 自动尝试安装 Chromium 运行时的系统依赖（容器里常缺）
#   - 自动选择国内镜像下载 Chromium（如果 PLAYWRIGHT_DOWNLOAD_HOST 未设置）
#   - 启动服务

set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# -------- 颜色输出 --------
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}[run]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[err]${NC} $*"; }

# -------- 0) 环境检测 --------
IN_CONTAINER=0
if [ -f /.dockerenv ] || grep -qE 'docker|kubepods|containerd' /proc/1/cgroup 2>/dev/null; then
    IN_CONTAINER=1
fi
HAS_ROOT=0
if [ "$(id -u 2>/dev/null)" = "0" ]; then
    HAS_ROOT=1
fi

if [ "$IN_CONTAINER" = "1" ]; then
    log "Detected: running inside a container."
else
    log "Detected: running on host machine."
fi

# -------- 1) Python 环境 --------
PY="python3"
command -v python3 >/dev/null 2>&1 || PY="python"

if [ "$IN_CONTAINER" = "1" ]; then
    # 容器内：直接用系统 Python，避免 venv 在某些精简镜像里失败
    log "Using system Python: $($PY --version)"
else
    if [ ! -d ".venv" ]; then
        log "Creating virtualenv .venv ..."
        $PY -m venv .venv
    fi
    # shellcheck source=/dev/null
    source .venv/bin/activate
    PY="python"
    log "Using virtualenv Python: $($PY --version)"
fi

# -------- 2) 选 pip 镜像（中国镜像加速） --------
PIP_INDEX_URL_DEFAULT="https://pypi.org/simple"
if [ -z "${PIP_INDEX_URL:-}" ]; then
    # 简单探测：能 ping 通 pypi.tuna.tsinghua.edu.cn 就用清华源
    if curl -fsSL --max-time 3 https://pypi.tuna.tsinghua.edu.cn/simple/ -o /dev/null 2>&1; then
        export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
        log "Using pip mirror: $PIP_INDEX_URL"
    fi
fi

# -------- 3) 装 Python 依赖 --------
log "Installing Python deps ..."
$PY -m pip install -q --upgrade pip
$PY -m pip install -q -r requirements.txt

# -------- 4) Playwright Chromium --------
# 如果用户没设镜像，容器内默认走 npmmirror（国内快很多）
if [ -z "${PLAYWRIGHT_DOWNLOAD_HOST:-}" ] && [ "$IN_CONTAINER" = "1" ]; then
    export PLAYWRIGHT_DOWNLOAD_HOST="https://npmmirror.com/mirrors/playwright"
    log "Using PLAYWRIGHT_DOWNLOAD_HOST=$PLAYWRIGHT_DOWNLOAD_HOST"
fi

# 检测 Chromium 是否已安装
if $PY -c "from playwright.sync_api import sync_playwright; \
import sys; \
p = sync_playwright().start(); \
sys.exit(0 if p.chromium.executable_path else 1); \
" 2>/dev/null; then
    log "Playwright Chromium already installed."
else
    log "Installing Playwright Chromium (~150MB) ..."
    $PY -m playwright install chromium
fi

# -------- 5) 安装 Chromium 系统库（容器内常缺） --------
install_system_deps() {
    if ! command -v apt-get >/dev/null 2>&1; then
        warn "apt-get not found, skip system deps. If chromium fails to start, install libs manually."
        return
    fi

    SUDO=""
    [ "$HAS_ROOT" = "0" ] && SUDO="sudo"

    log "Installing Chromium runtime libs (apt) ..."
    if [ -n "$SUDO" ] && ! command -v sudo >/dev/null 2>&1; then
        warn "no sudo and not root, skip apt install"
        return
    fi

    $SUDO apt-get update -qq || warn "apt update failed (network?)"
    $SUDO apt-get install -y --no-install-recommends \
        libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
        libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
        libpango-1.0-0 libcairo2 libasound2 libdrm2 \
        fonts-liberation ca-certificates 2>/dev/null \
        || warn "some system deps may be missing, continue anyway"
}

# 通过尝试启动 chromium 探测系统库是否齐全
log "Checking Chromium runtime libs ..."
if ! $PY - <<'PYEOF' 2>/dev/null
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    b.close()
PYEOF
then
    warn "Chromium failed to launch, trying to install system libs ..."
    install_system_deps || true
    # 再试一次
    if ! $PY - <<'PYEOF' 2>/dev/null
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    b.close()
PYEOF
    then
        err "Chromium still cannot launch."
        err "Please install missing libraries manually, hint: ldd \$($PY -m playwright install --dry-run chromium 2>&1 | grep -o '/[^ ]*chrome' | head -1) | grep 'not found'"
        exit 1
    fi
fi
log "Chromium runtime OK."

# -------- 6) 启动服务 --------
log "Starting browser service on ${HOST:-0.0.0.0}:${PORT:-8080} ..."
exec $PY -m app.main
