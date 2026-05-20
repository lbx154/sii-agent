# 联网搜索工具（Serper + Jina，支持代理穿透）

两个工具：
- `search_text(query, top_k=5, fetch=True, max_chars=5000)` — 文搜文（Google → 正文）
- `search_image(image, top_k=5, fetch=True, max_chars=5000)` — 图搜文（Google Lens → 正文）

返回结构统一：
```json
[
  {"rank": 1, "title": "...", "url": "...", "snippet": "...", "content": "...(markdown)..."}
]
```

---

## 一、部署架构

我们的 GPU 服务器上能跑模型但**没外网**；CPU 服务器有外网但**不能跑模型**。  
为此搜索工具支持两种工作模式：

```
┌─────────────────────────────┐                ┌──────────────────────────────┐
│ GPU server (no internet)    │                │ CPU server (has internet)    │
│                             │   HTTP over    │                              │
│  task_runner ─► search_tool │── SSH tunnel ──► search-proxy (FastAPI)       │
│        ▲                    │   (VS Code     │     │                        │
│        │ SEARCH_PROXY_URL=  │    forwarded   │     ├─► google.serper.dev    │
│        │ http://127.0.0.1:8090            )  │     ├─► r.jina.ai            │
│                             │                │     └─► 0x0.st (only for     │
│                             │                │            local image upload)│
└─────────────────────────────┘                └──────────────────────────────┘
```

GPU 端的 `search_tool.py` 看到 `SEARCH_PROXY_URL` 不为空就自动走代理模式；GPU 上**不需要任何 API key、不需要外网**，所有 key 都放在 CPU 这边。

> 留了一个直连模式作为兜底：当 `SEARCH_PROXY_URL` 不设置时仍按老版本直接打 Serper / Jina，方便本地开发或临时切回。

---

## 二、CPU 服务器端：起 search-proxy

```bash
# 在 CPU 服务器上
cd /path/to/harness/search-proxy

# 必填
export SERPER_API_KEY="your_key_here"
# 可选
export JINA_API_KEY="your_key_here"
# 强烈建议设一个随机字符串当做共享密钥
export PROXY_API_TOKEN="$(openssl rand -hex 16)"

./run.sh
# 默认监听 127.0.0.1:8090
```

> ⚠️ **故意只监听 127.0.0.1**。我们要走 SSH 隧道，没有理由把它放公网。

健康检查（在 CPU 服务器上）：
```bash
curl http://127.0.0.1:8090/health
# {"status":"ok","serper_configured":true,"jina_configured":true}
```

---

## 三、GPU 服务器端：打通端口

> 拿不到 CPU 服务器的内网 IP？看完这一节就有答案。先做一个判断再选方案。

### 0. 先确认 GPU 能不能出公网

```bash
# 在 GPU 服务器上
TUNNEL_URL="<把 VS Code / Codespaces 给你的转发链接贴这里>"
curl -sS -o /dev/null -w "%{http_code}\n" "$TUNNEL_URL/health" --max-time 10
```

- 返回 `200` / `401` / `403` → 走 **方案 A**（最省事）。
- 超时 / DNS 失败 → 跳到 **方案 D**（反向 SSH，最稳兜底）。

### 方案 A —— 直接用 VS Code / Codespaces 给的转发链接

如果 GPU 能解析并访问 `*.tunnel.vscode.dev`、`*.preview.app.github.dev`、`*.trycloudflare.com`、`*.ngrok-free.app` 这类公网链接，那根本不需要 SSH 隧道，直接把 `SEARCH_PROXY_URL` 设成这条链接就行：

```bash
# GPU 服务器上
export SEARCH_PROXY_URL="https://abc-8090.tunnel.vscode.dev"
export SEARCH_PROXY_TOKEN="..."     # 与 CPU 端 PROXY_API_TOKEN 一致

# 自签证书 / cloudflared quick tunnel 之类的，关掉 SSL 校验
# export SEARCH_PROXY_VERIFY_SSL=false

# 有的隧道（GitHub Codespaces / VS Code Tunnel 私有可见性）需要带 cookie：
# export SEARCH_PROXY_EXTRA_HEADERS='{"Cookie":"vscode-tkn=xxxx"}'

python -m tools.search_tool text "hello world" --no-fetch
```

**两个常见坑**：

1. **VS Code Tunnel / Codespaces 默认要求登录**：转发出来的链接默认 visibility 是 `Private`，浏览器直接打开会被强制走 GitHub 登录，curl 直接 401。
   - VS Code → `Ports` 面板 → 8090 端口 → 右键 → `Port Visibility → Public`。
   - 这样链接就变公开，工具和 curl 才能直接 POST 进去。注意此时**强烈建议**把 CPU 端的 `PROXY_API_TOKEN` 设成一个长随机串当兜底鉴权。

2. **转发链接是 HTTPS** —— 我们的 CPU 端服务监听的还是 plain HTTP `127.0.0.1:8090`，VS Code 网关会帮你做 TLS 终端，CPU 端无需改任何东西。

### 方案 B —— SSH 正向隧道（GPU 主动连 CPU，能拿到 CPU 内网 IP / 域名时用）

```bash
# 在 GPU 服务器上
ssh -N -L 127.0.0.1:8090:127.0.0.1:8090 user@cpu-host

# 然后 GPU 上访问 http://127.0.0.1:8090 等同于访问 CPU 的 127.0.0.1:8090
export SEARCH_PROXY_URL=http://127.0.0.1:8090
```

建议用 `autossh` 守住断线重连：
```bash
autossh -M 0 -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8090:127.0.0.1:8090 user@cpu-host
```

### 方案 C —— VS Code Tunnel 把 CPU 主机注册到你账号

CPU 端跑：
```bash
code tunnel --name cpu-proxy
# 跟着提示用 GitHub 登录一次
```

之后从你日常 VS Code 远程主机列表里就能选到 `cpu-proxy`，并把 8090 端口 forward 出来。**注意**：这个 forward 出来的链接最终也是 `*.tunnel.vscode.dev`，所以本质还是回到方案 A，仍然需要 GPU 能解析 `vscode.dev`。

### 方案 D —— 反向 SSH 隧道（**拿不到 IP / GPU 完全无外网时的首选兜底**）

由 **CPU 服务器主动**连到 GPU，把 8090 端口"塞"过去：

```bash
# 在 CPU 服务器上
ssh -N -R 127.0.0.1:8090:127.0.0.1:8090 user@gpu-host
# 或者用 autossh 守住
autossh -M 0 -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -R 127.0.0.1:8090:127.0.0.1:8090 user@gpu-host
```

- **不需要 CPU 的 IP**：用域名或 `~/.ssh/config` 配置的 Host alias 就行。
- **不需要 GPU 出公网**：流量走的是 CPU → GPU 这条 SSH 通道，跟那条 `vscode.dev` 链接完全无关。
- 多数发行版默认 sshd 就允许 loopback 反向转发；不行就在 GPU 的 `/etc/ssh/sshd_config` 加 `GatewayPorts clientspecified` 后 `systemctl reload sshd`。

之后 GPU 端只要：
```bash
export SEARCH_PROXY_URL=http://127.0.0.1:8090
```

---

## 四、GPU 服务器端：让 search_tool 走代理

```bash
# 在 GPU 服务器上 —— 不需要 SERPER_API_KEY / JINA_API_KEY
export SEARCH_PROXY_URL="http://127.0.0.1:8090"
export SEARCH_PROXY_TOKEN="刚才在CPU上设的同一个字符串"   # 如果开了鉴权

python -m tools.search_tool text "transformer 2025" --top-k 3
# 第一行会打印 [mode] proxy via http://127.0.0.1:8090
```

图搜文同样：
```bash
# 公网 URL 直接传
python -m tools.search_tool image "https://upload.wikimedia.org/.../Cat.jpg"

# 本地文件 —— 工具会把图片 multipart 上传到 CPU 端的 /upload_image
# 由 CPU 代你转传到 0x0.st，最后再把得到的公网 URL 喂给 Lens
python -m tools.search_tool image /home/user/photo.jpg
```

---

## 五、环境变量速查

GPU 端：

| 变量 | 必需？ | 说明 |
| --- | --- | --- |
| `SEARCH_PROXY_URL` | 推荐设置 | `http://127.0.0.1:8090`（隧道）或 `https://abc-8090.tunnel.vscode.dev`（公网转发链接）。设了就走代理；不设就走直连模式 |
| `SEARCH_PROXY_TOKEN` | 看 CPU 端是否开鉴权 | 与 CPU 端 `PROXY_API_TOKEN` 必须一致 |
| `SEARCH_PROXY_TIMEOUT` | 可选 | 单次 HTTP 调用秒数，默认 120 |
| `SEARCH_PROXY_VERIFY_SSL` | 可选 | 默认 `true`；遇到自签证书（cloudflared 等）设 `false` |
| `SEARCH_PROXY_EXTRA_HEADERS` | 可选 | JSON 字符串，例如 `'{"Cookie":"vscode-tkn=..."}'`，给私有 tunnel 带鉴权用 |
| `SERPER_API_KEY` / `JINA_API_KEY` | 仅直连模式 | 走代理时不需要 |

CPU 端（`search-proxy`）：

| 变量 | 必需？ | 说明 |
| --- | --- | --- |
| `SERPER_API_KEY` | ✅ | https://serper.dev 的 API Key |
| `JINA_API_KEY`   | 可选 | 提升 QPS / 减少限流 |
| `PROXY_API_TOKEN` | 可选 | 设了之后所有请求都要 `Authorization: Bearer <token>` |
| `HOST` / `PORT`  | 可选 | 默认 `127.0.0.1:8090` |
| `IMAGE_UPLOADER` | 可选 | 默认 `0x0`，仅 `/upload_image` 用 |

---

## 六、独立测试 / 自检

CPU 端：
```bash
curl -s http://127.0.0.1:8090/health
curl -s -X POST http://127.0.0.1:8090/search/text \
  -H "Authorization: Bearer $PROXY_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query":"hello world","top_k":2,"fetch":false}' | jq
```

GPU 端（隧道打通后）：
```bash
curl -s http://127.0.0.1:8090/health   # 应该返回 ok
python -m tools.search_tool text "foo" --no-fetch
```

---

## 七、设计要点 & 注意事项

1. **Key 收敛在 CPU 端**：GPU 上彻底无 key、无外网，运维口径更干净；CPU 端 `search-proxy` 才是唯一外网 egress 点。
2. **接口语义保持向后兼容**：GPU 端 `search_text` / `search_image` 的入参/返回结构不变，LLM 工具 schema 不需要任何修改。
3. **错误兜底**：代理调用失败时，工具不再抛异常，而是返回一条 `[proxy-error] ...` 的伪结果，让 LLM 自己决定要不要重试或换关键词。
4. **图片上传**：本地图片不再上传到 0x0.st 直接出公网，而是先 `multipart/form-data` 上传到 CPU 端的 `/upload_image`，由 CPU 代为转储。这样 GPU 上不需要任何外网 DNS。
5. **隧道运维**：建议用 `autossh` + systemd（或 `tmux`）守住隧道；隧道断了 `search_tool` 会立即返回连接错误，但不会 crash 整个 task_runner。
6. **token 预算**：默认 `top_k=5` × `max_chars=5000` ≈ 25K 字符。配合现在 `traj.to_messages()` 每轮带完整历史，几轮就会逼近 128K，与 `task_runner.py` 的上下文压缩问题一并处理时记得把 search 结果纳入截断目标。
