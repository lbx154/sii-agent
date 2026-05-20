"""演示通过 CDP URL 直接用 Playwright 连接（高级用法）。

适合需要完整 Playwright API 控制能力的场景。
注意：CDP 端口默认只在浏览器服务所在主机的 9222 上，
跨主机访问需要确保该端口同样在内网可达（默认监听 0.0.0.0）。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.async_api import async_playwright  # noqa: E402

from client import BrowserClient  # noqa: E402


async def main(base_url: str) -> None:
    bc = BrowserClient(base_url)
    info = bc.cdp_url()
    print("CDP info:", info)
    cdp_url = info["cdp_url"]

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        ctx = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = await ctx.new_page()
        await page.goto("https://example.com")
        print("title:", await page.title())
        await page.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    asyncio.run(main(url))
