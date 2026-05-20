# 在容器内运行 browser-service

针对 `agent-summer` 这种**已经在容器里**的环境（无 Docker 权限、可能没 systemd）。

## 🚀 一键启动（推荐先试这个）

```bash
cd /path/to/browser-service
chmod +x run.sh
./run.sh
```

`run.sh` 会自动：
- 识别容器环境，跳过 venv 直接用系统 Python
- 自动选用国内 pip 源（清华）+ Playwright 镜像（npmmirror）
- 检测并安装 Chromium 二进制
- 探测 Chromium 系统库是否齐全，缺则用 `apt-get` 自动补
- 启动服务监听 `0.0.0.0:8080`

如果一切顺利，你会看到：
```
[run] Starting browser service on 0.0.0.0:8080 ...
INFO:     Uvicorn running on http://0.0.0.0:8080
```

## 🧰 手动步骤（脚本失败时按步骤排查）

### 1) 装 Python 依赖

```bash
pip install -r requirements.txt
# 慢的话：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2) 装 Chromium 二进制

```bash
# 国内加速
export PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright
python -m playwright install chromium
```

### 3) 装 Chromium 运行所需系统库

容器（尤其精简镜像）经常缺这些。需要 root 或 sudo：

```bash
apt-get update
apt-get install -y --no-install-recommends \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libdrm2 \
    fonts-liberation ca-certificates
```

### 4) 启动

```bash
python -m app.main
```

## 🔌 让另一台机器访问

容器里的 `0.0.0.0:8080` **能不能从外面访问**取决于容器是怎么启动的：

| 情况 | 现象 | 解决 |
|------|------|------|
| 容器有独立 IP（同网段） | 直接 `curl http://容器IP:8080/health` 通 | 直接用 |
| 容器端口被宿主机映射（`-p 8080:8080`） | `curl http://宿主机IP:8080` 通 | 用宿主机 IP |
| 都不通 | 容器在隔离网络里 | **用 VSCode 端口转发**（见下） |

### VSCode 端口转发（最稳）

如果你用 VSCode Remote 连这个容器：

1. 启动服务 `./run.sh`，监听 8080
2. 看 VSCode 底部 **PORTS** 面板（没有就 `View → Ports` 打开）
3. VSCode 会自动检测到 8080
4. 右键 → `Port Visibility` →
   - **Private**：只有你登录账号能访问，更安全
   - **Public**：拿到 URL 的人都能访问（**务必配 API_TOKEN**）
5. 复制地址栏的 🌐 → 得到形如 `https://abc-8080.use.devtunnels.ms` 的公网 URL
6. 在另一台机器上：
   ```python
   from browser_client import BrowserClient
   bc = BrowserClient(
       "https://abc-8080.use.devtunnels.ms",
       token="你的API_TOKEN",   # 强烈建议
   )
   bc.health()
   ```

### 启用 API_TOKEN（公网暴露时必做）

```bash
# 容器内
cp .env.example .env
echo 'API_TOKEN=随便一串长字符串_xxx_yyy_zzz' >> .env
# 重启服务
```

客户端调用：
```python
BrowserClient(url, token="随便一串长字符串_xxx_yyy_zzz")
```

## ❓ 容器内常见报错

### `error while loading shared libraries: libnss3.so`
缺系统库，回到 [步骤 3](#3-装-chromium-运行所需系统库) 装一下。

### `Failed to launch chromium because executable doesn't exist`
没装 Chromium 二进制，跑：
```bash
python -m playwright install chromium
```

### `Permission denied` 启动 Chromium
普通容器需要 `--no-sandbox`（项目代码里已加）。如果还报错，检查 seccomp 限制：
```bash
# 临时排查：看进程能不能用 unshare
unshare --user echo ok 2>&1
```
通常解决不了，得让运维放权或换环境。

### `apt-get: command not found`（Alpine 镜像）
Alpine 用 `apk`，命令不一样：
```bash
apk add --no-cache nss freetype harfbuzz ca-certificates ttf-freefont \
    chromium    # Alpine 直接用系统 Chromium
```
然后改 `app/browser.py` 的 `launch()` 加 `executable_path="/usr/bin/chromium-browser"`。

### Playwright 下载 Chromium 卡 / 失败
```bash
export PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright
# 或
export PLAYWRIGHT_DOWNLOAD_HOST=https://playwright.azureedge.net    # 官方
python -m playwright install chromium --force
```

### 内存不足 OOM
容器内存太小（< 1GB）跑 Chromium 容易崩。建议至少 1GB，多页面建议 2GB。

## 🩺 一键诊断脚本

```bash
echo "=== 环境 ===" 
cat /proc/1/comm
[ -f /.dockerenv ] && echo "in docker" || echo "not docker"
uname -a
cat /etc/os-release 2>/dev/null | head -3

echo "=== Python ==="
python3 --version
python3 -c "import playwright; print('playwright', playwright.__version__)" 2>&1

echo "=== Chromium ==="
python3 -c "from playwright.sync_api import sync_playwright; \
p=sync_playwright().start(); \
print('chromium path:', p.chromium.executable_path)" 2>&1

echo "=== 启动测试 ==="
python3 -c "from playwright.sync_api import sync_playwright; \
with sync_playwright() as p: \
    b=p.chromium.launch(headless=True, args=['--no-sandbox','--disable-dev-shm-usage']); \
    print('launched ok'); b.close()" 2>&1 | tail -5
```

把输出贴出来，能很快定位卡在哪。
