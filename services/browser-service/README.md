# Browser Service

一个**轻量级浏览器 HTTP 服务**：在你的本机（Mac/Linux）启动一个 Headless Chromium，通过 HTTP API 暴露给同一内网的其他服务器调用，让它们可以远程操纵浏览器进行导航、获取文本、截图、点击、执行 JS 等。

> 灵感来自 [agent-infra/sandbox](https://github.com/agent-infra/sandbox)，但去掉 VNC/VSCode/Jupyter/MCP，**只保留浏览器**，纯 Python，无需 Docker。

---

## ✨ 功能

- ✅ **Headless** Chromium（Playwright 内核）
- ✅ **多 session 隔离** - cookie/storage 互不影响，自动空闲回收
- ✅ **多 tab 管理** - 单 session 可开多个标签页
- ✅ **完整浏览器操作** - 导航、文本、HTML、截图、点击、输入、滚动、JS
- ✅ **CDP 直连** - 暴露 Chromium 的 `9222` 端口，可用 Playwright/Puppeteer 直接接管
- ✅ **可选 Bearer Token 鉴权**
- ✅ **自动 Swagger 文档**：http://host:8080/docs
- ✅ **纯 Python**，一键脚本启动

---

## 🚀 快速开始

### 1. 启动服务（在你 Mac 上）

```bash
cd /Users/george_young/Downloads/browser-service
chmod +x run.sh
./run.sh
```

首次启动会：
- 创建 `.venv` 虚拟环境
- 安装 Python 依赖
- 下载 Playwright Chromium（约 150MB，仅首次）
- 启动服务监听 `0.0.0.0:8080`

启动成功后，访问：
- **API 文档**：http://localhost:8080/docs
- **健康检查**：http://localhost:8080/health

### 2. 让内网服务器访问

#### 2.1 查看 Mac 在内网的 IP

```bash
# Mac
ipconfig getifaddr en0     # Wi-Fi
ipconfig getifaddr en1     # 有线
```

假设 IP 是 `192.168.1.10`。

#### 2.2 macOS 防火墙放行（首次启动时通常会自动弹窗，点"允许"）

如果没弹：**系统设置 → 网络 → 防火墙 → 选项 → 允许 Python 接入**。

#### 2.3 在内网服务器上测试

```bash
curl http://192.168.1.10:8080/health
```

返回 `{"status":"ok",...}` 即代表打通。

### 3. 用 Python SDK 调用

把 `client/browser_client.py` 复制到内网服务器，或：

```bash
# 内网服务器
pip install requests
```

```python
from browser_client import BrowserClient

bc = BrowserClient("http://192.168.1.10:8080")

# 创建会话
sid = bc.create_session()["session_id"]

# 导航
bc.navigate("https://example.com", session_id=sid)

# 获取文本
text = bc.get_text(session_id=sid)
print(text[:200])

# 获取标题里的链接
links = bc.eval_js(
    "() => Array.from(document.querySelectorAll('a')).map(a => a.href)",
    session_id=sid,
)
print(links)

# 截图
bc.screenshot(session_id=sid, save_to="page.png", full_page=True)

# 关闭
bc.close_session(sid)
```

---

## 📚 API 一览

完整 OpenAPI 文档：http://localhost:8080/docs

### Session / Tab

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/session/create` | 新建 session（含一个默认 tab） |
| GET | `/session/list` | 列出所有 session |
| DELETE | `/session/{session_id}` | 关闭 session |
| POST | `/tab/new` | 在 session 里新开 tab |
| POST | `/tab/close` | 关闭 tab |
| GET | `/tab/list/{session_id}` | 列出 session 的所有 tab |

### 浏览器操作

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/browser/navigate` | 打开 URL |
| POST | `/browser/get_text` | 取页面/元素文本 |
| POST | `/browser/get_html` | 取页面/元素 HTML |
| POST | `/browser/screenshot` | 截图（base64） |
| POST | `/browser/click` | 点击元素 |
| POST | `/browser/type` | 输入文本（可回车） |
| POST | `/browser/scroll` | 滚动页面 |
| POST | `/browser/eval` | 执行 JS |
| GET | `/browser/title` | 当前页面 title 与 URL |
| GET | `/browser/cdp_url` | 拿到 Chromium 的 CDP URL |

### 请求 Body 通用字段

绝大多数 POST 接口都接受：
- `session_id` - 不传则自动用第一个 session（导航接口会自动建一个）
- `tab_id` - 不传则用 session 的活动 tab
- `selector` - CSS 选择器（部分接口可选）

---

## ⚙️ 配置

复制 `.env.example` 为 `.env` 自定义：

```bash
cp .env.example .env
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8080` | 监听端口 |
| `HEADLESS` | `true` | 无界面模式 |
| `BROWSER_CDP_HOST` | `127.0.0.1` | Chromium DevTools 监听地址；如需跨主机 CDP 再显式设为 `0.0.0.0` |
| `BROWSER_CDP_PORT` | `9222` | Chromium DevTools 端口 |
| `DEFAULT_VIEWPORT_WIDTH` | `1280` | 视口宽 |
| `DEFAULT_VIEWPORT_HEIGHT` | `800` | 视口高 |
| `MAX_SESSIONS` | `10` | 同时存在的最大 session 数 |
| `MAX_TABS_PER_SESSION` | `20` | 每个 session 的最大 tab 数，超过后回收最旧 tab |
| `SESSION_IDLE_TIMEOUT` | `1800` | session 空闲多少秒后自动回收 |
| `BROWSER_BLOCK_RESOURCE_TYPES` | （空） | 可选 Playwright 资源类型屏蔽列表，如 `font,media` |
| `API_TOKEN` | （空） | 设置后所有接口都需 `Authorization: Bearer xxx` |

---

## 🔌 高级：CDP 直连（用 Playwright/Puppeteer 接管）

```python
from playwright.async_api import async_playwright
from browser_client import BrowserClient

cdp = BrowserClient("http://192.168.1.10:8080").cdp_url()["cdp_url"]
# → http://127.0.0.1:9222 （服务所在主机视角）

# 跨主机访问 CDP：把 cdp 中的 127.0.0.1 替换成 Mac 的内网 IP
cdp = cdp.replace("127.0.0.1", "192.168.1.10")

async with async_playwright() as p:
    browser = await p.chromium.connect_over_cdp(cdp)
    ...
```

> ⚠️ 如果把 `BROWSER_CDP_HOST` 设为 `0.0.0.0`，务必只在**可信内网**暴露，否则等于把整个浏览器开放给外网。

---

## 🛠️ 项目结构

```
browser-service/
├── README.md
├── requirements.txt
├── .env.example
├── run.sh                  # 一键启动
├── app/
│   ├── main.py             # FastAPI 入口
│   ├── config.py           # 配置
│   ├── browser.py          # 浏览器 + 会话管理
│   ├── routes.py           # HTTP 路由
│   └── schemas.py          # 请求/响应模型
├── client/
│   └── browser_client.py   # Python SDK
└── examples/
    ├── demo.py             # 基本调用演示
    └── cdp_direct.py       # 通过 CDP 直连 Playwright
```

---

## ❓ 常见问题

**Q: 启动时报 `playwright install` 网络慢？**  
A: 国内用户可设置环境变量再跑 `run.sh`：
```bash
export PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright
./run.sh
```

**Q: 内网调不通？**  
A: 依次排查：
1. Mac 端 `lsof -i :8080` 确认服务在监听
2. Mac 防火墙是否放行 Python
3. 内网服务器 `curl -v http://<mac-ip>:8080/health` 看是连接被拒还是超时
4. 是否同一网段（公司/家庭 Wi-Fi 有时分隔了不同 VLAN）

**Q: 想关掉 headless 看看浏览器？**  
A: 改 `.env`：`HEADLESS=false`，重启服务即可（前提是有图形环境）。

**Q: 资源占用？**  
A: 单浏览器实例约 200-400MB 内存，每个 session/tab 增加几十 MB。`MAX_SESSIONS` 限制并发上限。

---

## 📄 License

参考上游 sandbox 项目，沿用 Apache-2.0。
