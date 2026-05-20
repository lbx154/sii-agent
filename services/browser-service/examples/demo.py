"""调用浏览器服务的演示脚本。

在内网另一台服务器上运行：
    pip install requests
    python demo.py http://<mac-ip>:8080
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保能 import 同级 client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client import BrowserClient  # noqa: E402


def main(base_url: str) -> None:
    bc = BrowserClient(base_url)
    print("health:", bc.health())

    s = bc.create_session()
    sid = s["session_id"]
    print("session:", sid)

    print("→ 打开 example.com")
    print(bc.navigate("https://example.com", session_id=sid))

    print("→ 页面文本：")
    print(bc.get_text(session_id=sid)[:300])

    print("→ 标题：", bc.title(session_id=sid))

    print("→ 截图存到 ./demo.png")
    bc.screenshot(session_id=sid, full_page=True, save_to="./demo.png")

    print("→ 执行 JS 取所有链接：")
    links = bc.eval_js(
        "() => Array.from(document.querySelectorAll('a')).map(a => a.href)",
        session_id=sid,
    )
    print(links)

    print("→ 关闭 session")
    bc.close_session(sid)
    print("done.")


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    main(url)
