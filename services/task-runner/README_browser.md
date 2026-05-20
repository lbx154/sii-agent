# 浏览器工具（browser-service + Playwright）

5 个工具，全部对 LLM 暴露，全部返回 dict / list[dict]，失败统一为 `{ok: false, error: "..."}`：

| 工具 | 输入要点 | 主要返回字段 |
| --- | --- | --- |
| `browser_navigate(url, ...)` | URL（自动补 https:// 前缀） | `ok, url, title, wait_until, text_preview?, truncated?` |
| `browser_get_text(max_chars, timeout)` | — | `ok, url, title, text, truncated, total_chars` |
| `browser_click(selector, nth, timeout)` | CSS 选择器；`nth>0` 时通过 JS 命中第 N 个匹配 | `ok, selector, current_url, current_title, navigated` |
| `browser_type(selector, text, submit, clear, timeout)` | CSS 选择器 + 文本 | `ok, selector, submitted, current_url, current_title` |
| `browser_parallel(urls, mode, ...)` | URL 列表，mode∈{navigate,get_text} | `list[dict]`（每条同 navigate / get_text 的字段） |

> 对外签名 / 返回结构与之前的 AIO-Sandbox 版本**完全保持一致**，所以 LLM 的工具 schema 不需要任何改动。

---

## 一、运行机制

```
LLM ──tool call──▶ tools/browser_tool.py            (host)
                         │
                         ▼
                   sandbox_client.BrowserSandboxClient
                         │ (HTTP, 单例 + 默认 session_id)
                         ▼
                   browser-service (FastAPI on Mac)
                         │
                         ▼
                   Headless Chromium (Playwright)
```

设计要点：
- **不再注入脚本到沙盒里**，也不再依赖 `agent_sandbox`。每次调用就是一个普通的 HTTP 请求。
- **会话状态跨工具调用保留**：`BrowserSandboxClient` 持有一个默认 `session_id`，所有同步工具（`navigate / get_text / click / type`）都复用同一个 tab，与之前的语义一致。
- **`browser_parallel` 用 ThreadPoolExecutor + 多 tab 实现**：每个 URL 临时新开一个 tab，结束即关；不影响"当前 tab"。
- **错误规范化**：把 HTTP 4xx/5xx 都翻译成 `{ok: False, error: "..."}`，避免异常往 LLM 那边漏。

---

## 二、配置

环境变量：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `SANDBOX_BASE_URL` | `http://localhost:8080` | browser-service 的入口地址 |
| `SANDBOX_API_TOKEN` | （空） | 如果服务端开了 Bearer 鉴权就设这个 |
| `SANDBOX_HTTP_TIMEOUT` | `120` | 单次 HTTP 请求的默认超时（秒）|

依赖：
```bash
pip install -r requirements.txt
```

启动浏览器服务（在你 Mac 上）：
```bash
cd /Users/george_young/Downloads/browser-service
./run.sh
# 监听 0.0.0.0:8080，Swagger 文档：http://localhost:8080/docs
```

如果 harness 跑在另一台机器，把 `SANDBOX_BASE_URL` 指到 Mac 的内网 IP，例如：
```bash
export SANDBOX_BASE_URL=http://192.168.1.10:8080
```

---

## 三、CLI 自测

```bash
# 单页
python -m tools.browser_tool navigate https://example.com
python -m tools.browser_tool get_text
python -m tools.browser_tool click "a[href='/more']"
python -m tools.browser_tool type "input[name=q]" "hello world" --submit

# 并发（默认 mode=navigate）
python -m tools.browser_tool parallel https://a.com https://b.com https://c.com
python -m tools.browser_tool parallel https://a.com https://b.com --mode get_text
```

---

## 四、与上一版（AIO Sandbox）对比

| 维度 | 上一版（agent_sandbox） | 当前版（browser-service） |
| --- | --- | --- |
| 通讯方式 | 通过 `sb.shell.exec_command` 在沙盒里运行 Python dispatcher，stdout 框 sentinel JSON | 直接 HTTP 调用 FastAPI，JSON 进 / JSON 出 |
| 脚本注入 | `_pw_runner.py` 上传到 `/tmp`，按 sha256 缓存 | **无脚本注入**，已删除 `_pw_runner.py` |
| Playwright 依赖 | 沙盒里懒装（首次 60-120s）| 服务端预装；客户端无需 Playwright |
| 错误回包 | `{ok:false, error}` | `{ok:false, error}`（保持一致）|
| 会话语义 | 复用 `contexts[0].pages[0]` | 客户端持有默认 `session_id`，后端复用对应 tab |
| 并发实现 | 沙盒内 asyncio + 多 page | 客户端 `ThreadPoolExecutor` + 多 tab |
| `nth` 点击 | Playwright `locator.nth()` | 没有原生接口，回退到 JS：`document.querySelectorAll(sel)[n].click()` |

---

## 五、注意事项 / 已知坑

1. **页面状态在多轮工具调用之间是保留的**：所有 sync 工具默认操作同一个 `session_id` 的活动 tab。`browser_parallel` 会临时新开 tab，结束即关。
2. **CSS 选择器要 LLM 自己写**：服务端没有 snapshot 接口；LLM 可以先 `browser_get_text` 看页面再写选择器。
3. **`wait_until` 取值**：`load | domcontentloaded | networkidle | commit`。其它值会被规整成 `domcontentloaded`。
4. **session 空闲回收**：`browser-service` 默认 `SESSION_IDLE_TIMEOUT=1800s`，长任务里要注意 keep-alive；如果遇到 404（session 不存在），调用 `sandbox_client.reconnect_sandbox()` 即可重建。
5. **鉴权**：服务端开了 `API_TOKEN` 时，harness 这边设置 `SANDBOX_API_TOKEN` 即可，客户端会自动加 `Authorization: Bearer ...`。
6. **nth 点击的小限制**：当 `nth>0` 时走的是 JS `click()`，不会触发某些只对真实鼠标事件响应的 hover/focus 逻辑。常见场景已经够用，遇到反例再考虑扩展。
