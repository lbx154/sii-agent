"""
Browser tool — drive a remote Chromium via the *browser-service* HTTP API.

Public functions (signatures and return shapes are kept identical to the
previous AIO-Sandbox-based implementation so the LLM tool schema does not
need to change):

    browser_navigate(url, ...)
    browser_get_text(...)
    browser_click(selector, ...)
    browser_type(selector, text, ...)
    browser_parallel(urls, mode, ...)

Each function returns a ``dict`` (or ``list[dict]`` for ``browser_parallel``).
On any failure the result is ``{"ok": False, "error": "..."}`` — no exception
is propagated to the caller.

Implementation strategy
-----------------------
The new ``browser-service`` already exposes a clean HTTP API. We simply call
it through ``BrowserSandboxClient`` (see ``sandbox_client.py``) which keeps a
single default ``session_id`` for the whole task — meaning successive tool
calls act on the same browser tab, just like before.

For ``browser_parallel`` we spin up a fresh tab per URL and dispatch them
across a small ``ThreadPoolExecutor``; the leftover tabs are closed at the
end so the original session's "current tab" is unchanged.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


def _load_tool_env() -> None:
    explicit = os.getenv("HARNESS_APIKEY_FILE", "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend(parent / "apikey.md" for parent in Path(__file__).resolve().parents)
    seen: set[Path] = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            try:
                parts = shlex.split(raw, comments=True, posix=True)
            except ValueError:
                continue
            if parts and parts[0] == "export":
                parts = parts[1:]
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key) and not os.getenv(key):
                    os.environ[key] = value
        return


_load_tool_env()

from sandbox_client import BrowserSandboxClient, get_sandbox

logger = logging.getLogger("harness.tools.browser")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_RE_BLANK_LINES = re.compile(r"\n{3,}")
_RE_INLINE_WS   = re.compile(r"[ \t]{2,}")


def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = _RE_BLANK_LINES.sub("\n\n", s)
    s = _RE_INLINE_WS.sub(" ", s)
    return s.strip()


def _truncate(s: str, max_chars: int) -> tuple[str, bool]:
    if max_chars and len(s) > max_chars:
        return s[:max_chars], True
    return s, False


def _normalize_url(url: str) -> str:
    if not url:
        return url
    if url.startswith(("http://", "https://", "file://", "data:")):
        return url
    if url in ("about:blank", "about:newtab"):
        return url
    return "https://" + url


def _normalize_wait_until(wait_until: str) -> str:
    """browser-service accepts: load | domcontentloaded | networkidle | commit."""
    valid = {"load", "domcontentloaded", "networkidle", "commit"}
    return wait_until if wait_until in valid else "domcontentloaded"


def _err(msg: str, **extra: Any) -> dict:
    out = {"ok": False, "error": msg}
    out.update(extra)
    return out


def _safe_get_text(
    cli: BrowserSandboxClient,
    selector: Optional[str] = None,
    tab_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """Return (text, error_or_None). Selector defaults to whole-body."""
    try:
        text = cli.get_text(selector=selector, tab_id=tab_id)
        return _clean_text(text), None
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {exc}"


def _safe_title(cli: BrowserSandboxClient, tab_id: Optional[str] = None) -> dict:
    try:
        return cli.title(tab_id=tab_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("title() failed: %s", exc)
        return {"title": "", "url": ""}


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------
def browser_navigate(
    url: str,
    wait_until: str = "domcontentloaded",
    include_text: bool = True,
    max_text: int = 2000,
    timeout: int = 30,
) -> dict:
    """Navigate the browser to ``url`` (default tab of the default session).

    Returns:
        {ok, url, title, wait_until, text_preview?, truncated?, text_error?}
        On failure: {ok: False, error: "..."}
    """
    if not url or not url.strip():
        return _err("url is empty")

    real_url = _normalize_url(url.strip())
    wu = _normalize_wait_until(wait_until)
    timeout_ms = max(1, int(timeout)) * 1000

    logger.info("browser_navigate url=%s wait=%s", real_url, wu)
    try:
        cli = get_sandbox()
        nav = cli.navigate(real_url, wait_until=wu, timeout_ms=timeout_ms)
    except Exception as exc:  # noqa: BLE001
        return _err(f"navigate failed: {type(exc).__name__}: {exc}")

    out: dict = {
        "ok":         True,
        "url":        nav.get("url", real_url),
        "title":      nav.get("title", ""),
        "wait_until": wu,
    }
    if include_text:
        text, text_err = _safe_get_text(cli)
        if text_err is None:
            txt, truncated = _truncate(text, int(max_text))
            out["text_preview"] = txt
            out["truncated"]    = truncated
        else:
            out["text_preview"] = ""
            out["text_error"]   = text_err
    return out


def browser_get_text(max_chars: int = 5000, timeout: int = 15) -> dict:
    """Return cleaned visible text from the current page.

    Returns:
        {ok, url, title, text, truncated, total_chars}
        On failure: {ok: False, error: "..."}
    """
    logger.info("browser_get_text max_chars=%d", int(max_chars))
    try:
        cli = get_sandbox()
        text, text_err = _safe_get_text(cli)
        if text_err is not None:
            return _err(text_err)
        meta = _safe_title(cli)
    except Exception as exc:  # noqa: BLE001
        return _err(f"get_text failed: {type(exc).__name__}: {exc}")

    txt, truncated = _truncate(text, int(max_chars))
    return {
        "ok":          True,
        "url":         meta.get("url", ""),
        "title":       meta.get("title", ""),
        "text":        txt,
        "truncated":   truncated,
        # ``total_chars`` is reported pre-truncation so the LLM knows how big
        # the real page is.
        "total_chars": len(text),
    }


def browser_click(selector: str, nth: int = 0, timeout: int = 10) -> dict:
    """Click an element matching a CSS selector.

    NOTE: the underlying service always clicks ``locator(selector).first``;
    the legacy ``nth`` parameter is therefore implemented by rewriting the
    selector with ``:nth-of-type`` semantics via JS ``eval`` when ``nth>0``.
    For ``nth==0`` we use the simple click endpoint.
    """
    if not selector or not selector.strip():
        return _err("selector is empty")
    timeout_ms = max(1, int(timeout)) * 1000
    nth = int(nth)

    logger.info("browser_click selector=%r nth=%d", selector, nth)
    try:
        cli = get_sandbox()
        before = _safe_title(cli)

        if nth > 0:
            # Service has no first-class nth, do it via JS to keep semantics.
            click_script = (
                "(args) => { const els = document.querySelectorAll(args.sel);"
                " if (!els || els.length <= args.n) { return false; }"
                " els[args.n].click(); return true; }"
            )
            ok = cli.eval_js(
                f"({click_script})({{sel: {selector!r}, n: {nth}}})"
            )
            if not ok:
                return _err(f"selector {selector!r} matched <= {nth} elements")
        else:
            cli.click(selector, timeout_ms=timeout_ms)

        # Best-effort wait for any navigation triggered by the click.
        try:
            cli.eval_js(
                "() => new Promise(r => "
                "document.readyState === 'complete' ? r(true) : "
                "window.addEventListener('load', () => r(true), {once:true}))"
            )
        except Exception:  # noqa: BLE001
            pass

        after = _safe_title(cli)
    except Exception as exc:  # noqa: BLE001
        return _err(f"click failed: {type(exc).__name__}: {exc}")

    return {
        "ok":            True,
        "selector":      selector,
        "current_url":   after.get("url", ""),
        "current_title": after.get("title", ""),
        "navigated":     bool(after.get("url") and after.get("url") != before.get("url")),
    }


def browser_type(
    selector: str,
    text: str,
    submit: bool = False,
    clear: bool = True,
    timeout: int = 10,
) -> dict:
    """Type text into an element matching a CSS selector.

    Args:
        submit: if True, press Enter after typing.
        clear:  if True, clear the field before typing.
    """
    if not selector or not selector.strip():
        return _err("selector is empty")
    if text is None:
        return _err("text is None")

    text_str = str(text)
    logger.info(
        "browser_type selector=%r submit=%s len=%d", selector, submit, len(text_str)
    )

    try:
        cli = get_sandbox()
        cli.type_text(
            selector=selector,
            text=text_str,
            clear=bool(clear),
            press_enter=bool(submit),
            # 20ms/key matches the previous behaviour (locator.type(delay=20)).
            delay_ms=20,
        )
        meta = _safe_title(cli)
    except Exception as exc:  # noqa: BLE001
        return _err(f"type failed: {type(exc).__name__}: {exc}")

    _ = timeout  # accepted for API compatibility; service uses its own defaults
    return {
        "ok":            True,
        "selector":      selector,
        "submitted":     bool(submit),
        "current_url":   meta.get("url", ""),
        "current_title": meta.get("title", ""),
    }


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------
def _parallel_one(
    cli: BrowserSandboxClient,
    url: str,
    mode: str,
    max_chars: int,
    wait_until: str,
    timeout_ms: int,
) -> dict:
    """Open ``url`` in a fresh tab, extract data, then close the tab."""
    tab_id = ""
    try:
        new = cli.new_tab()
        tab_id = new.get("tab_id", "")
        if not tab_id:
            return {"ok": False, "url": url, "error": "new_tab returned empty tab_id"}

        cli.navigate(url, wait_until=wait_until, timeout_ms=timeout_ms, tab_id=tab_id)
        meta = _safe_title(cli, tab_id=tab_id)
        title = meta.get("title", "")
        final_url = meta.get("url", url)

        text, text_err = _safe_get_text(cli, tab_id=tab_id)

        if mode == "navigate":
            if text_err is not None:
                return {
                    "ok":          True,
                    "url":         final_url,
                    "title":       title,
                    "text_preview": "",
                    "text_error":   text_err,
                }
            txt, truncated = _truncate(text, max_chars)
            return {
                "ok":           True,
                "url":          final_url,
                "title":        title,
                "text_preview": txt,
                "truncated":    truncated,
            }

        # mode == "get_text"
        if text_err is not None:
            return {"ok": False, "url": final_url, "error": text_err}
        txt, truncated = _truncate(text, max_chars)
        return {
            "ok":          True,
            "url":         final_url,
            "title":       title,
            "text":        txt,
            "truncated":   truncated,
            "total_chars": len(text),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if tab_id:
            try:
                cli.close_tab(tab_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("close_tab(%s) failed: %s", tab_id, exc)


def browser_parallel(
    urls: list,
    mode: str = "navigate",
    max_chars: Optional[int] = None,
    wait_until: str = "domcontentloaded",
    max_concurrency: int = 4,
    timeout: int = 30,
) -> list:
    """Open multiple URLs concurrently, each in its own tab.

    Args:
        urls:    list of URLs.
        mode:    "navigate" -> {url,title,text_preview,truncated} per item
                 "get_text" -> {url,title,text,truncated,total_chars} per item
        max_chars: per-item character cap. Defaults: 2000 for navigate, 5000 for get_text.
        max_concurrency: simultaneous tabs (1-8).

    Returns:
        list[dict]; per-item ``{ok: False, url, error}`` on individual failure.
        On host-level failure returns ``[{"ok": False, "error": "..."}]``.
    """
    if not urls:
        return [_err("urls list is empty")]
    if mode not in ("navigate", "get_text"):
        return [_err(f"unsupported mode: {mode}")]

    cleaned = [_normalize_url(u.strip()) for u in urls if u and u.strip()]
    if not cleaned:
        return [_err("no valid urls after normalization")]

    if max_chars is None:
        max_chars = 2000 if mode == "navigate" else 5000
    concurrency = max(1, min(int(max_concurrency), 8))
    wu = _normalize_wait_until(wait_until)
    timeout_ms = max(1, int(timeout)) * 1000

    logger.info(
        "browser_parallel n=%d mode=%s conc=%d", len(cleaned), mode, concurrency
    )

    try:
        cli = get_sandbox()
        cli.ensure_session()
    except Exception as exc:  # noqa: BLE001
        return [_err(f"sandbox unavailable: {type(exc).__name__}: {exc}")]

    results: list[dict] = [None] * len(cleaned)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_idx = {
            pool.submit(
                _parallel_one,
                cli,
                u,
                mode,
                int(max_chars),
                wu,
                timeout_ms,
            ): i
            for i, u in enumerate(cleaned)
        }
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[i] = {
                    "ok": False,
                    "url": cleaned[i],
                    "error": f"{type(exc).__name__}: {exc}",
                }
    return results


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pprint

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("navigate"); p1.add_argument("url")
    sub.add_parser("get_text")
    p3 = sub.add_parser("click");    p3.add_argument("selector")
    p4 = sub.add_parser("type")
    p4.add_argument("selector"); p4.add_argument("text"); p4.add_argument("--submit", action="store_true")
    p5 = sub.add_parser("parallel")
    p5.add_argument("urls", nargs="+")
    p5.add_argument("--mode", default="navigate", choices=["navigate", "get_text"])

    args = ap.parse_args()
    if args.cmd == "navigate":
        pprint.pp(browser_navigate(args.url))
    elif args.cmd == "get_text":
        pprint.pp(browser_get_text())
    elif args.cmd == "click":
        pprint.pp(browser_click(args.selector))
    elif args.cmd == "type":
        pprint.pp(browser_type(args.selector, args.text, submit=args.submit))
    elif args.cmd == "parallel":
        pprint.pp(browser_parallel(args.urls, mode=args.mode))
