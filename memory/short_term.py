"""Per-attempt working memory for compact evidence tracking.

This memory is intentionally ephemeral: it is rebuilt for each ReAct run,
never persisted, and never consumes gold labels. It summarizes tool usage so
long tool trajectories keep a compact view of what has already been tried.
"""
from __future__ import annotations

import json
import re
from collections import deque
from typing import Any


_DOCID_RE = re.compile(r'"docid"\s*:\s*"([^"]+)"')
_TITLE_RE = re.compile(r"--- title:\s*([^\n]+)")
_RANKED_RE = re.compile(r"^\[(\d+)\]\s+(.+)$")


def _squash(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 12].rstrip() + " ...[cut]"


def _json_arg(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


class ShortTermMemory:
    """Compact working memory for a single agent attempt."""

    def __init__(self, question: str, max_items: int = 12, max_chars: int = 2500):
        self.question = question
        self.max_chars = max(400, int(max_chars))
        self.queries: deque[str] = deque(maxlen=max_items)
        self.evidence: deque[str] = deque(maxlen=max_items)
        self.dead_ends: deque[str] = deque(maxlen=max_items)
        self.notices: deque[str] = deque(maxlen=max_items)
        self.no_tool_notes: deque[str] = deque(maxlen=4)
        self.observations = 0
        self.injections = 0
        self.injected_chars = 0

    def observe_no_tool(self, content: str) -> None:
        if content.strip():
            self.no_tool_notes.append(_squash(content, 220))

    def observe_tool(self, name: str, arguments: dict[str, Any], result: str) -> None:
        self.observations += 1
        result_text = str(result or "")
        lowered = result_text.lower()

        query = _json_arg(arguments, "query") or _json_arg(arguments, "url") or _json_arg(arguments, "source")
        if query:
            self.queries.append(f"{name}: {_squash(query, 220)}")

        if (
            not result_text.strip()
            or "(no results)" in lowered
            or "error:" in lowered
            or "timeout" in lowered
            or "notice:" in lowered
            or "unavailable" in lowered
        ):
            self.dead_ends.append(f"{name}({ _squash(query, 120) if query else '?' }) -> {_squash(result_text, 260)}")
            return

        if name in {"search", "get_document"}:
            self._observe_json_or_doc_evidence(name, result_text)
        elif name in {"wiki_search", "web_search"}:
            self._observe_ranked_evidence(name, result_text)
        elif name in {"browser_text", "browser_open", "browser_open_many"}:
            self._observe_page_evidence(name, result_text)
        else:
            self.notices.append(f"{name}: {_squash(result_text, 320)}")

    def _observe_json_or_doc_evidence(self, name: str, text: str) -> None:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        if items:
            for item in items[:3]:
                if not isinstance(item, dict):
                    continue
                docid = str(item.get("docid") or item.get("id") or "?")
                snippet = str(item.get("snippet") or item.get("text") or "")[:600]
                title_match = _TITLE_RE.search(snippet)
                title = f" title={_squash(title_match.group(1), 120)}" if title_match else ""
                self.evidence.append(f"{name} docid={docid}{title}: {_squash(snippet, 260)}")
            return
        docids = _DOCID_RE.findall(text)
        if docids:
            self.evidence.append(f"{name} docids seen: {', '.join(docids[:6])}")
        else:
            self.evidence.append(f"{name}: {_squash(text, 320)}")

    def _observe_ranked_evidence(self, name: str, text: str) -> None:
        kept = 0
        for raw_line in text.splitlines():
            line = raw_line.strip()
            match = _RANKED_RE.match(line)
            if not match:
                continue
            self.evidence.append(f"{name} hit {match.group(1)}: {_squash(match.group(2), 260)}")
            kept += 1
            if kept >= 3:
                break
        if kept == 0:
            self.evidence.append(f"{name}: {_squash(text, 320)}")

    def _observe_page_evidence(self, name: str, text: str) -> None:
        title = ""
        url = ""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                title = str(data.get("title") or "")
                url = str(data.get("url") or "")
                body = str(data.get("text") or data.get("content") or "")
                self.evidence.append(f"{name} page {title or url}: {_squash(body, 320)}")
                return
        except json.JSONDecodeError:
            pass
        for prefix in ("TITLE:", "URL:"):
            idx = text.find(prefix)
            if idx >= 0:
                fragment = text[idx: idx + 240]
                self.evidence.append(f"{name}: {_squash(fragment, 320)}")
                return
        self.evidence.append(f"{name}: {_squash(text, 320)}")

    def render_for_prompt(self) -> str:
        if not (self.queries or self.evidence or self.dead_ends or self.no_tool_notes or self.notices):
            return ""
        sections: list[str] = ["[Working memory for this attempt — compact evidence only, not a source of truth]"]
        if self.queries:
            sections.append("Queries/targets already tried:\n" + "\n".join(f"- {q}" for q in self.queries))
        if self.evidence:
            sections.append("Useful evidence snippets/docids seen:\n" + "\n".join(f"- {e}" for e in self.evidence))
        if self.dead_ends:
            sections.append("Dead ends/errors to avoid repeating:\n" + "\n".join(f"- {d}" for d in self.dead_ends))
        if self.no_tool_notes:
            sections.append("Prior no-tool assistant text:\n" + "\n".join(f"- {n}" for n in self.no_tool_notes))
        if self.notices:
            sections.append("Other observations:\n" + "\n".join(f"- {n}" for n in self.notices))
        sections.append("Use the current tool outputs as authority; use this memory only to avoid loops and preserve compact evidence.")
        rendered = "\n\n".join(sections)
        rendered = _squash(rendered, self.max_chars)
        self.injections += 1
        self.injected_chars += len(rendered)
        return rendered

    def stats(self) -> dict[str, int]:
        return {
            "observations": self.observations,
            "injections": self.injections,
            "injected_chars": self.injected_chars,
        }
