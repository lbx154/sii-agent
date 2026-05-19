"""Minimal ReAct loop with OpenAI-style tool calling, governed by a Harness."""
from __future__ import annotations
import json
import os
import re
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from .llm import chat
from tools import TOOL_REGISTRY, tool_specs, dispatch
from tools.runtime_context import set_tool_context
from harness.controller import HarnessConfig, HarnessResult, StepGuard
from memory.short_term import ShortTermMemory

BENCHMARK_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browsecomp_search",
    "browsecomp_open",
    "browser_open",
    "browser_open_many",
    "final_answer",
)
VISUAL_TOOLS = (
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "web_search",
    "wiki_search",
    "wiki_page",
    "reverse_image_search",
    "browser_open",
    "browser_open_many",
    "final_answer",
)
RICH_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "reverse_image_search",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "browser_close",
    "final_answer",
)
MEMORY_TOOLS = (
    "web_search",
    "wiki_search",
    "wiki_page",
    "browsecomp_search",
    "browsecomp_open",
    "browser_open",
    "browser_open_many",
    "memory_search",
    "memory_stats",
    "memory_list",
    "memory_get",
    "memory_create",
    "memory_update",
    "memory_delete",
    "final_answer",
)
WRITABLE_MEMORY_TOOLS = {"memory_create", "memory_update", "memory_delete"}
TRAIN_ONLY_TOOLS = WRITABLE_MEMORY_TOOLS
TOOL_PROFILES = {
    "benchmark": BENCHMARK_TOOLS,
    "default": BENCHMARK_TOOLS,
    "visual": VISUAL_TOOLS,
    "rich": RICH_TOOLS,
    "full": RICH_TOOLS,
    "memory": MEMORY_TOOLS,
    "self_retrieval": MEMORY_TOOLS,
}
DEFAULT_TOOLS = BENCHMARK_TOOLS
_BROWSER_EXTRACT_TOOLS = {"browser_open", "browser_open_many", "browser_text", "browser_click", "browser_type"}
_RESEARCH_TOOLS = {
    "web_search",
    "wiki_search",
    "wiki_page",
    "browser_open",
    "browser_open_many",
    "browser_text",
    "browser_click",
    "browser_type",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "reverse_image_search",
    "browsecomp_search",
    "browsecomp_open",
}
_EVIDENCE_SUMMARY_PREFIX = "EVIDENCE_STATE_UPDATE:"
_COMPACT_CONTEXT_PREFIX = "COMPACT_CONTEXT_UPDATE:"
_EVIDENCE_SUMMARY_IMMEDIATE_TOOLS = {
    "browser_open",
    "browser_open_many",
    "browser_text",
    "visual_web_search",
    "image_to_text",
    "image_to_search_queries",
    "reverse_image_search",
    "browsecomp_open",
}
_TOOL_NAME_ALIASES = {
    "web_open": "browser_open",
    "web_open_url": "browser_open",
    "browser_open_url": "browser_open",
    "open_url": "browser_open",
    "search": "browsecomp_search",
    "get_document": "browsecomp_open",
    "browsecomp_get_document": "browsecomp_open",
    "browsecomp_get": "browsecomp_open",
}


SYSTEM_PROMPT = """You are a careful research agent.

Loop:
1. Think briefly about what you still need.
2. Call exactly ONE available tool.
3. Read the tool result and decide next step.

Available tools for this run:
{tool_list}

Rules:
{tool_rules}
- Cross-check at least one source for non-trivial claims. If the user provides a `Provided context` section, treat that context as the primary source and do not search only to cross-check facts already present there.
- DO NOT repeat the same query / URL — refine instead.
- For distinctive phrases or unusual clue wording, run an early exact quoted-phrase search instead of only broad paraphrases.
- For hard web-research questions, do not stop at the first plausible candidate. Keep searching until the answer-specific constraint is verified or the tool budget is exhausted.
- If a browser result is blocked, redirected to an unrelated domain, or lacks the requested answer field, prefer a targeted `web_search` for the exact entity plus answer-field phrase before switching entities.
- Never put uncertainty, explanations, or "not found" narratives in `final_answer.answer`; it must be only the best concise answer span.
- For person names, copy the full source-backed name exactly, including all given names and particles/casing (e.g. do not shorten a multi-token first name).
- Answer in the same language as the user question when possible; for Chinese questions, prefer the common Chinese name/phrase over an English alias.
- Use only as many evidence tool calls as needed, but verify the final answer span. Training memory CRUD calls are allowed when the rules require them.
- When you have a plausible answer and have completed required memory actions, call `final_answer` with a concise answer.
- You MUST call `final_answer` before the step budget is exhausted.
- If stuck, simplify the query or pivot to a related search term.
"""


def _shell_tool_enabled() -> bool:
    return os.getenv("SII_AGENT_ENABLE_SHELL_TOOL", "").strip().lower() in {"1", "true", "yes"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _runtime_mode() -> str:
    mode = os.getenv("SII_AGENT_RUNTIME_MODE", "train").strip().lower()
    return mode if mode in {"train", "test"} else "train"


def _profile_tools() -> tuple[str, ...]:
    profile = os.getenv("SII_AGENT_TOOL_PROFILE", "benchmark").strip().lower()
    if profile == "all":
        tools = tuple(TOOL_REGISTRY)
    else:
        tools = TOOL_PROFILES.get(profile, BENCHMARK_TOOLS)
    tools = tuple(name for name in tools if name != "verify")
    if _runtime_mode() == "test":
        tools = tuple(name for name in tools if name not in TRAIN_ONLY_TOOLS)
    if _shell_tool_enabled() and profile in {"self_retrieval", "memory", "rich", "full", "all"} and "bash_exec" not in tools:
        tools = (*tools, "bash_exec")
    if not _shell_tool_enabled():
        tools = tuple(name for name in tools if name != "bash_exec")
    return tools


def _tool_rules(allowed_tools: tuple[str, ...]) -> str:
    tools = set(allowed_tools)
    rules: list[str] = []
    if "memory_search" in tools:
        rules.append(
            "- Early in the task, generate 2-6 focused query phrases yourself (entities, relation words, task pattern, likely failure mode), then call `memory_search` with both the full question and `queries`. It returns compressed actionable guidance plus compact record ids; use memory_get only when you need full content for update/delete."
        )
    if "memory_search" in tools and _runtime_mode() == "test":
        rules.append("- Runtime mode is `test`: memory is read-only; retrieve useful records but do not create, update, or delete memory.")
    if "memory_create" in tools:
        rules.append(
            "- You are in train mode: improving memory is part of the task, not optional bookkeeping. "
            "Do not wait for the final step: before `final_answer`, call `memory_create` when this question teaches a reusable lesson or procedural skill. "
            "If `memory_search` exposed stale, wrong, duplicate, or harmful memory, call `memory_update` or `memory_delete` on that concrete id. "
            "For internal training-check rejections, do not create failure/reflection memory immediately; revise the answer first, and the harness will persist recovery memory only if the corrected answer verifies. "
            "Skip CRUD when the retrieved memory is good and the current run teaches nothing reusable."
        )
    if "web_search" in tools:
        rules.append("- Use `web_search` for live facts through the configured search-proxy or direct Serper fallback; it returns top snippets, not full pages.")
    if "wiki_search" in tools:
        rules.append("- Use `wiki_search` and `wiki_page` for offline encyclopedic facts when available.")
    if "browsecomp_search" in tools:
        rules.append(
            "- Use `browsecomp_search` for text-only BrowseComp-style web-research questions; it searches the local fixed corpus and often beats live web search for obscure clue chains. Start with distinctive quoted phrases or rare entity/constraint terms, then refine."
        )
    if "browsecomp_open" in tools:
        rules.append("- Use `browsecomp_open` to retrieve a full local BrowseComp document by docid returned from `browsecomp_search`.")
    if "browser_open" in tools:
        rules.append(
            "- Use `browser_open` to read one promising URL; it reads the full page and returns query-focused excerpts when possible. "
            "If the returned title/url belongs to a different site than requested, treat it as a source mismatch, not as evidence for a new entity."
        )
    if "browser_open_many" in tools:
        rules.append("- Use `browser_open_many` to read several independent source URLs concurrently.")
    if "visual_web_search" in tools:
        rules.append(
            "- For visual factual questions with an image path or image URL, prefer `visual_web_search` first when available. It gathers evidence but does not produce the final answer."
        )
    if "image_to_search_queries" in tools or "image_to_text" in tools or "reverse_image_search" in tools:
        rules.append("- For image questions, combine visible/OCR clues with search evidence; do not treat the first visual entity guess as proven.")
    if "reverse_image_search" in tools:
        rules.append("- When calling `reverse_image_search`, pass the user question as `query` so it can fall back to text search if image upload/lens fails.")
    if "browser_click" in tools or "browser_type" in tools:
        rules.append("- Use browser interaction tools only when a page needs interaction after opening.")
    if "bash_exec" in tools:
        rules.append("- Use `bash_exec` only for explicit command-line inspection; prefer specialized search/memory/browser tools.")
    return "\n".join(rules) if rules else "- Use the available tools only when they directly help answer the question."


def _system_prompt(allowed_tools: tuple[str, ...], extra_system: str | None) -> str:
    sys = SYSTEM_PROMPT.format(
        tool_list=", ".join(f"`{name}`" for name in allowed_tools),
        tool_rules=_tool_rules(allowed_tools),
    )
    if "memory_search" in set(allowed_tools):
        overall = _overall_memory_guidance()
        if overall:
            sys += f"\n\n[Overall Memory Guidance]\n{overall}"
    if _env_bool("SII_AGENT_EVIDENCE_SUMMARY", False):
        sys += (
            "\n\n[Evidence State]\n"
            "Maintain a compact constraint table: requested answer type, required constraints, candidate entities, "
            "satisfied/missing/contradicted evidence, and the next targeted query. If an `EVIDENCE_STATE_UPDATE` "
            "appears, treat it as a working synthesis, not independent evidence. Preserve high-confidence candidates "
            "unless a source-backed contradiction shows they fail; never switch entities only because a page redirected."
        )
    if _env_bool("SII_AGENT_CONTEXT_COMPACT", False):
        sys += (
            "\n\n[Compact Context]\n"
            "If a `COMPACT_CONTEXT_UPDATE` appears, treat it as compressed notes from earlier steps, not independent "
            "evidence. Combine it with the latest raw tool results and evidence state; if they conflict, prefer raw "
            "source-backed observations and re-check the constraint."
        )
    if extra_system:
        sys += f"\n\n[Memory & Reflection]\n{extra_system}"
    return sys


def _clip_prompt_text(text: object, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value[:limit]


def _overall_memory_guidance() -> str:
    if os.getenv("SII_MEMORY_OVERALL_IN_PROMPT", "1").strip().lower() in {"0", "false", "no"}:
        return ""
    root = Path(os.getenv("SII_AGENT_MEMORY_ROOT", os.getenv("MEMORY_ROOT", "logs/memory")))
    path = root / "overall.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, dict):
        return ""
    try:
        configured_limit = int(os.getenv("SII_MEMORY_OVERALL_PROMPT_MAX_CHARS", "1600"))
    except ValueError:
        configured_limit = 1600
    max_chars = max(400, min(2400, configured_limit))
    lines = [
        "Use this as process guidance only; it is not evidence for the current answer.",
        _clip_prompt_text(data.get("overall"), max_chars),
    ]
    categories = data.get("categories")
    if isinstance(categories, list):
        for item in categories[:6]:
            if not isinstance(item, dict):
                continue
            pattern = _clip_prompt_text(item.get("pattern"), 80)
            guidance = _clip_prompt_text(item.get("guidance"), 220)
            if pattern and guidance:
                lines.append(f"- {pattern}: {guidance}")
    avoid = data.get("avoid")
    if isinstance(avoid, list) and avoid:
        lines.append("Avoid: " + "; ".join(_clip_prompt_text(item, 140) for item in avoid[:4] if item))
    rendered = "\n".join(line for line in lines if line).strip()
    return _clip_prompt_text(rendered, max_chars)


def _internal_gold_verify_enabled(expected: str | None) -> bool:
    if _runtime_mode() != "train" or expected is None:
        return False
    return os.getenv("SII_AGENT_ENABLE_GOLD_VERIFY", "1").strip().lower() not in {"0", "false", "no"}


def _browser_extract_query(question: str) -> str:
    text = str(question or "").strip()
    match = re.search(r"(?:^|\n)\s*Question:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1)
    text = re.split(
        r"\n\s*\n|\s+Use the relevance-ranked\b|\s+Use the relevant\b|\s+Provided context\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return " ".join(text.split())[:1000]


def _with_browser_extract_query(name: str, args: dict[str, Any], question: str) -> dict[str, Any]:
    if name not in _BROWSER_EXTRACT_TOOLS or str(args.get("extract_query") or "").strip():
        return args
    extract_query = _browser_extract_query(question)
    if not extract_query:
        return args
    updated = dict(args)
    updated["extract_query"] = extract_query
    return updated


def _tool_count(counts: dict[str, int], names: set[str]) -> int:
    return sum(int(counts.get(name) or 0) for name in names)


def _safe_json(s: str) -> dict | None:
    try:
        return json.loads(s)
    except Exception:
        return None


def _parse_json_object_from_text(text: str) -> dict | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        raw = fenced.group(1)
    if raw.startswith("{"):
        parsed = _safe_json(raw)
        if isinstance(parsed, dict):
            return parsed
        try:
            parsed_literal = ast.literal_eval(raw)
            if isinstance(parsed_literal, dict):
                return parsed_literal
        except Exception:  # noqa: BLE001
            pass
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index, ch in enumerate(raw[start:], start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start:index + 1]
                parsed = _safe_json(candidate)
                if isinstance(parsed, dict):
                    return parsed
                try:
                    parsed_literal = ast.literal_eval(candidate)
                    return parsed_literal if isinstance(parsed_literal, dict) else None
                except Exception:  # noqa: BLE001
                    return None
    return None


def _response_text(response: Any) -> str:
    try:
        msg = response.choices[0].message
    except Exception:  # noqa: BLE001
        return ""
    content = str(getattr(msg, "content", "") or "")
    if content.strip():
        return content
    reasoning = _message_reasoning(msg)
    return str(reasoning or "")


def _json_key_value_from_text(text: str, key: str) -> str:
    pattern = re.compile(rf"""["']{re.escape(key)}["']\s*:\s*""", flags=re.IGNORECASE)
    decoder = json.JSONDecoder()
    for match in pattern.finditer(text or ""):
        tail = text[match.end():].lstrip()
        if not tail:
            continue
        if tail[0] == '"':
            try:
                value, _end = decoder.raw_decode(tail)
                return str(value)
            except Exception:  # noqa: BLE001
                quoted = re.match(r'"((?:\\.|[^"\\])*)"', tail, flags=re.DOTALL)
                if quoted:
                    try:
                        return str(json.loads(f'"{quoted.group(1)}"'))
                    except Exception:  # noqa: BLE001
                        return quoted.group(1)
        if tail[0] == "'":
            quoted = re.match(r"'((?:\\.|[^'\\])*)'", tail, flags=re.DOTALL)
            if quoted:
                return quoted.group(1)
        bare = re.match(r"([^,\n}\]]+)", tail)
        if bare:
            return bare.group(1).strip().strip('"\'')
    return ""


def _clean_text_final_answer(text: str) -> str:
    value = " ".join(str(text or "").split()).strip()
    match = re.match(r"(?is)^(?:final\s+answer|answer)\s*[:：]\s*(.+)$", value)
    if match:
        value = match.group(1).strip()
    return value


def _extract_final_answer_args_from_text(content: str) -> dict[str, str] | None:
    text = str(content or "").strip()
    if not text:
        return None
    lower = text.lower()
    if "final_answer" not in lower and "<function=final_answer" not in lower:
        return None
    answer = _clean_text_final_answer(_json_key_value_from_text(text, "answer"))
    if not answer:
        return None
    rationale = _json_key_value_from_text(text, "rationale")
    return {"answer": answer, "rationale": rationale}


def _coerce_parameter_value(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    parsed = _safe_json(text)
    if parsed is not None:
        return parsed
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.fullmatch(r"-?\d+\.\d+", text):
        try:
            return float(text)
        except ValueError:
            return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    return text


def _payload_to_tool_call(payload: Any, call_id: str) -> Any | None:
    if not isinstance(payload, dict):
        return None
    function = payload.get("function") if isinstance(payload.get("function"), dict) else {}
    name = str(payload.get("name") or function.get("name") or "").strip()
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = payload.get("parameters")
    if arguments is None:
        arguments = function.get("arguments")
    if not name or arguments is None:
        return None
    if isinstance(arguments, str):
        arguments_text = arguments
    else:
        arguments_text = json.dumps(arguments, ensure_ascii=False)
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments_text),
        type="function",
        _synthetic=True,
    )


def _parse_content_tool_calls(content: str) -> list[Any]:
    calls: list[Any] = []
    seen_spans: list[tuple[int, int]] = []
    parsed_content = _safe_json((content or "").strip())
    if isinstance(parsed_content, dict):
        parsed_items = parsed_content.get("tool_calls") if isinstance(parsed_content.get("tool_calls"), list) else [parsed_content]
    elif isinstance(parsed_content, list):
        parsed_items = parsed_content
    else:
        parsed_items = []
    for payload in parsed_items:
        call = _payload_to_tool_call(payload, f"content-tool-call-{len(calls)}")
        if call is not None:
            calls.append(call)
    for i, match in enumerate(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL)):
        payload = _safe_json(match.group(1))
        call = _payload_to_tool_call(payload, f"content-tool-call-{len(calls)}")
        if call is None:
            continue
        calls.append(call)
        seen_spans.append(match.span())
    qwen_pattern = re.compile(
        r"<tool_call>\s*<function=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*?)</function>\s*</tool_call>",
        re.DOTALL,
    )
    for match in qwen_pattern.finditer(content or ""):
        if any(start <= match.start() < end for start, end in seen_spans):
            continue
        name = match.group(1).strip()
        body = match.group(2)
        args: dict[str, Any] = {}
        for param in re.finditer(r"<parameter=([A-Za-z_][A-Za-z0-9_-]*)>\s*(.*?)\s*</parameter>", body, re.DOTALL):
            args[param.group(1).strip()] = _coerce_parameter_value(param.group(2))
        calls.append(
            SimpleNamespace(
                id=f"content-tool-call-{len(calls)}",
                function=SimpleNamespace(name=name, arguments=json.dumps(args, ensure_ascii=False)),
                type="function",
                _synthetic=True,
            )
        )
    if not any(str(call.function.name).strip().lower() == "final_answer" for call in calls):
        final_args = _extract_final_answer_args_from_text(content or "")
        if final_args is not None:
            calls.append(
                SimpleNamespace(
                    id=f"content-tool-call-{len(calls)}",
                    function=SimpleNamespace(name="final_answer", arguments=json.dumps(final_args, ensure_ascii=False)),
                    type="function",
                    _synthetic=True,
                )
            )
    return calls


def _split_inline_reasoning(content: str) -> tuple[str, str]:
    text = content or ""
    reasoning_parts = [
        match.group(1).strip()
        for match in re.finditer(r"<think\b[^>]*>(.*?)</think>", text, flags=re.I | re.S)
        if match.group(1).strip()
    ]
    visible = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.I | re.S).strip()
    return visible, "\n\n".join(reasoning_parts)


def _canonical_tool_name(name: str, active_tools: tuple[str, ...]) -> str:
    raw = str(name or "").strip()
    lower = raw.lower()
    if lower in active_tools:
        return lower
    alias = _TOOL_NAME_ALIASES.get(lower)
    if alias in active_tools:
        return alias
    return raw


def _dump_tool_call(tc: Any, active_tools: tuple[str, ...] = ()) -> dict:
    canonical_name = _canonical_tool_name(tc.function.name, active_tools) if active_tools else tc.function.name
    if hasattr(tc, "model_dump"):
        dumped = tc.model_dump()
        function = dumped.get("function")
        if isinstance(function, dict):
            function["name"] = canonical_name
        return dumped
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": canonical_name,
            "arguments": tc.function.arguments,
        },
    }


def _message_reasoning(msg: Any) -> str:
    for name in ("reasoning_content", "reasoning"):
        value = getattr(msg, name, None)
        if value:
            return str(value)
    extra = getattr(msg, "model_extra", None)
    if isinstance(extra, dict):
        for name in ("reasoning_content", "reasoning"):
            value = extra.get(name)
            if value:
                return str(value)
    if hasattr(msg, "model_dump"):
        try:
            data = msg.model_dump()
        except Exception:  # noqa: BLE001
            data = {}
        if isinstance(data, dict):
            for name in ("reasoning_content", "reasoning"):
                value = data.get(name)
                if value:
                    return str(value)
    return ""


def _forced_final_tool_choice() -> dict:
    return {"type": "function", "function": {"name": "final_answer"}}


def _memory_auto_prefetch_enabled(active_tools: tuple[str, ...]) -> bool:
    return "memory_search" in active_tools and _env_bool("SII_MEMORY_AUTO_PREFETCH", True)


def _prefetch_memory_queries(question: str) -> list[str]:
    if not _env_bool("SII_MEMORY_AUTO_PREFETCH_EXTRA_QUERIES", False):
        return []
    text = _browser_extract_query(question)
    phrases: list[str] = []
    for quoted in re.findall(r'"([^"]{3,120})"|“([^”]{3,120})”|\'([^\']{3,120})\'', text):
        value = next((item for item in quoted if item), "")
        if value:
            phrases.append(value)
    words = re.findall(r"[A-Z][A-Za-z0-9'.-]*(?:\s+[A-Z][A-Za-z0-9'.-]*){1,4}", text)
    blocked = {
        "according", "answer", "august", "december", "everyone", "please", "question",
        "return", "specific", "there", "this", "what", "when", "where", "which", "who",
    }
    for word in words:
        clean_word = word.strip()
        if clean_word.lower() in blocked:
            continue
        phrases.append(clean_word)
    seen: set[str] = set()
    out: list[str] = []
    for phrase in phrases:
        clean = " ".join(phrase.split())[:120]
        key = clean.lower()
        if not clean or key in seen:
            continue
        out.append(clean)
        seen.add(key)
        if len(out) >= 4:
            break
    return out


def _auto_prefetch_memory(
    res: HarnessResult,
    messages: list[dict],
    question: str,
    user_content: Any | None,
    task: str | None,
) -> None:
    set_tool_context(question=question, user_content=user_content if user_content is not None else question, task=task, trajectory=[])
    args = {
        "query": question,
        "queries": _prefetch_memory_queries(question),
        "k": _env_int("SII_MEMORY_AUTO_PREFETCH_K", 6, 1, 20),
        "guidance_max_chars": _env_int("SII_MEMORY_AUTO_PREFETCH_GUIDANCE_CHARS", 900, 240, 1800),
        "auto_prefetch": True,
    }
    tool_result = dispatch("memory_search", args)
    prefetch_note = (
        "AUTOMATIC_MEMORY_PREFETCH_RESULT:\n"
        "This is read-only procedural guidance from memory, not evidence. "
        "Use it only as a search/verification plan and verify answers from current tools.\n"
        f"{tool_result}"
    )
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0].get('content', '')}\n\n{prefetch_note}"
    else:
        messages.insert(0, {"role": "system", "content": prefetch_note})
    res.tool_call_counts["memory_search_prefetch"] = res.tool_call_counts.get("memory_search_prefetch", 0) + 1
    _append_tool_event(res, "memory_search", args, tool_result, original_name="memory_search_prefetch")


def _apply_final_tool_call(res: HarnessResult, tc: Any) -> bool:
    if str(tc.function.name or "").strip().lower() != "final_answer":
        return False
    args = _safe_json(tc.function.arguments or "{}")
    if args is None:
        return False
    raw_answer = str(args.get("answer", ""))
    nested_args = _extract_final_answer_args_from_text(raw_answer)
    if nested_args is not None:
        raw_answer = nested_args.get("answer", "")
        if not str(args.get("rationale") or "").strip() and nested_args.get("rationale"):
            args["rationale"] = nested_args["rationale"]
    answer, _ = _split_inline_reasoning(raw_answer)
    answer = _clean_text_final_answer(answer)
    answer = answer.strip()
    if not answer or _looks_invalid_final_answer(answer):
        return False
    res.final_answer = answer
    rationale, _ = _split_inline_reasoning(str(args.get("rationale", "")))
    res.rationale = rationale
    res.stop_reason = "final"
    return True


def _looks_invalid_final_answer(answer: str) -> bool:
    text = " ".join(str(answer or "").split())
    if not text:
        return True
    lower = text.lower()
    non_answer_markers = (
        "the question asks",
        "the answer should be",
        "has not been found",
        "cannot be found",
        "not publicly available",
        "not indexed",
        "no search results",
        "unable to find",
        "i could not",
        "i cannot",
        "appears to be",
        "given the extensive search",
    )
    if any(marker in lower for marker in non_answer_markers):
        return True
    # BrowseComp-style answers are names, titles, dates, numbers, or short phrases.
    if len(text) > _env_int("SII_FINAL_ANSWER_MAX_CHARS", 160, 40, 500) and len(text.split()) > 18:
        return True
    return False


def _append_tool_event(
    res: HarnessResult,
    name: str,
    args: dict,
    content: str,
    *,
    original_name: str | None = None,
) -> None:
    event = {"role": "tool", "name": name, "args": args, "content": content}
    if original_name and original_name != name:
        event["original_name"] = original_name
    res.trajectory.append(event)


def _messages_with_short_memory(
    messages: list[dict],
    memory: ShortTermMemory | None,
    active_tools: tuple[str, ...],
) -> list[dict]:
    if memory is None:
        return messages
    rendered = memory.render_for_prompt()
    if not rendered:
        return messages
    memory_msg = {"role": "user", "content": rendered}
    if active_tools == ("final_answer",) and messages:
        return messages[:-1] + [memory_msg, messages[-1]]
    return messages + [memory_msg]


def _evidence_summary_enabled(active_tools: tuple[str, ...]) -> bool:
    if not _env_bool("SII_AGENT_EVIDENCE_SUMMARY", False):
        return False
    return any(name in set(active_tools) for name in _RESEARCH_TOOLS)


def _image_parts_for_summary(user_content: Any | None) -> list[dict[str, Any]]:
    if not _env_bool("SII_EVIDENCE_SUMMARY_WITH_IMAGE", True):
        return []
    if not isinstance(user_content, list):
        return []
    parts: list[dict[str, Any]] = []
    for part in user_content:
        if not isinstance(part, dict):
            continue
        if str(part.get("type") or "") == "image_url":
            parts.append(part)
        if len(parts) >= 1:
            break
    return parts


def _host_from_url(value: Any) -> str:
    try:
        parsed = urlparse(str(value or ""))
    except Exception:  # noqa: BLE001
        return ""
    return (parsed.netloc or "").lower().removeprefix("www.")


def _same_site_host(left: str, right: str) -> bool:
    if not left or not right:
        return True
    return left == right or left.endswith(f".{right}") or right.endswith(f".{left}")


def _looks_access_blocked(payload: dict[str, Any]) -> bool:
    haystack = " ".join(str(payload.get(key) or "") for key in ("title", "text", "raw")).lower()
    blocked_markers = (
        "undeclared automated tool",
        "access denied",
        "captcha",
        "bot detection",
        "automated traffic",
        "temporarily blocked",
    )
    return any(marker in haystack for marker in blocked_markers)


def _compact_tool_observation(name: str, args: dict[str, Any], content: str) -> dict[str, Any]:
    per_tool_chars = _env_int("SII_EVIDENCE_SUMMARY_TOOL_CHARS", 6000, 1000, 12000)
    compact_args = _clip_prompt_text(json.dumps(args, ensure_ascii=False, sort_keys=True), 800)
    parsed = _safe_json(content or "")
    if isinstance(parsed, dict):
        payload: dict[str, Any] = {}
        for key in ("title", "url", "source_type", "text_mode", "extract_query", "full_text_chars", "returned_text_chars", "omitted_text_chars"):
            if key in parsed:
                payload[key] = parsed[key]
        if "text" in parsed:
            payload["text"] = _clip_prompt_text(parsed.get("text"), per_tool_chars)
        if "links" in parsed and isinstance(parsed.get("links"), list):
            payload["links"] = parsed["links"][:8]
        if "results" in parsed and isinstance(parsed.get("results"), list):
            payload["results"] = parsed["results"][:4]
        if not payload:
            payload["raw"] = _clip_prompt_text(parsed, per_tool_chars)
        if name in _BROWSER_EXTRACT_TOOLS:
            requested_host = _host_from_url(args.get("url"))
            returned_host = _host_from_url(parsed.get("url"))
            if requested_host and returned_host and not _same_site_host(requested_host, returned_host):
                payload["source_warning"] = (
                    f"Requested host {requested_host}, but returned host {returned_host}. "
                    "Treat this as a redirect/source mismatch, not evidence for switching entities."
                )
            if _looks_access_blocked(payload):
                payload["access_warning"] = (
                    "The page appears blocked or anti-bot filtered. Do not treat missing fields as absence; "
                    "try a targeted web_search snippet or alternate source for the same entity."
                )
        compact_content: Any = payload
    else:
        compact_content = _clip_prompt_text(content, per_tool_chars)
    return {"tool": name, "args": compact_args, "observation": compact_content}


def _normalize_evidence_list(value: Any, limit: int, item_chars: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    out: list[Any] = []
    for item in value:
        if isinstance(item, dict):
            normalized = {
                str(key): _clip_prompt_text(val, item_chars)
                for key, val in item.items()
                if val is not None and str(val).strip()
            }
            if normalized:
                out.append(normalized)
        else:
            text = _clip_prompt_text(item, item_chars)
            if text:
                out.append(text)
        if len(out) >= limit:
            break
    return out


def _candidate_name(item: Any) -> str:
    if isinstance(item, dict):
        raw = item.get("name") or item.get("candidate") or item.get("entity") or item.get("answer") or ""
    else:
        raw = item
    text = re.sub(r"[^0-9A-Za-z\u00C0-\uFFFF]+", " ", str(raw or "").lower())
    return " ".join(text.split())


def _candidate_confidence(item: Any) -> str:
    if not isinstance(item, dict):
        return ""
    return str(item.get("confidence") or "").strip().lower()


def _substantive_candidate_field(value: Any) -> str:
    text = " ".join(str(value or "").split())
    lowered = text.lower().strip(" .;:-")
    if lowered in {"", "none", "n/a", "na", "unknown", "not applicable", "no conflicts", "none yet"}:
        return ""
    return text


def _harden_candidate_state(candidates: list[Any]) -> list[Any]:
    hardened: list[Any] = []
    for item in candidates:
        if not isinstance(item, dict):
            hardened.append(item)
            continue
        updated = dict(item)
        missing = _substantive_candidate_field(updated.get("missing"))
        conflicts = _substantive_candidate_field(updated.get("conflicts"))
        combined = f"{missing} {conflicts} {_substantive_candidate_field(updated.get('support'))}".lower()
        has_negative = any(
            marker in combined
            for marker in (
                "fails",
                "contradict",
                "not verified",
                "unverified",
                "not found",
                "no evidence",
                "does not",
                "mismatch",
            )
        )
        confidence = _candidate_confidence(updated)
        if conflicts or has_negative:
            updated["confidence"] = "low"
        elif missing and confidence == "high":
            updated["confidence"] = "medium"
        hardened.append(updated)
    return hardened


def _merge_candidate_continuity(previous: list[Any], current: list[Any]) -> list[Any]:
    if not previous or not current:
        return _harden_candidate_state(current or previous)
    current_names = {_candidate_name(item) for item in current}
    retained: list[Any] = []
    for item in previous:
        name = _candidate_name(item)
        if not name or name in current_names:
            continue
        if _candidate_confidence(item) in {"high", "medium"}:
            retained.append(item)
    if not retained:
        return _harden_candidate_state(current)
    merged = retained + current
    return _harden_candidate_state(merged[:8])


def _normalize_evidence_state(parsed: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    state = dict(previous or {})
    if parsed.get("summary") is not None:
        state["summary"] = _clip_prompt_text(parsed.get("summary"), 500)
    state["constraints"] = _normalize_evidence_list(parsed.get("constraints"), 10, 240) or state.get("constraints", [])
    previous_candidates = state.get("candidates", [])
    current_candidates = _normalize_evidence_list(parsed.get("candidates"), 8, 260)
    state["candidates"] = _merge_candidate_continuity(previous_candidates, current_candidates) or _harden_candidate_state(previous_candidates)
    state["verified_facts"] = _normalize_evidence_list(parsed.get("verified_facts"), 10, 240) or state.get("verified_facts", [])
    state["tentative_rejections"] = (
        _normalize_evidence_list(parsed.get("tentative_rejections"), 6, 220)
        or state.get("tentative_rejections", [])
    )
    state["next_searches"] = _normalize_evidence_list(parsed.get("next_searches"), 5, 180) or state.get("next_searches", [])
    state["pitfalls"] = _normalize_evidence_list(parsed.get("pitfalls"), 5, 180) or state.get("pitfalls", [])
    return state


def _render_evidence_state(state: dict[str, Any], max_chars: int | None = None) -> str:
    limit = max_chars or _env_int("SII_EVIDENCE_STATE_MAX_CHARS", 2200, 800, 6000)
    rendered = json.dumps(state or {}, ensure_ascii=False, indent=2)
    return _clip_prompt_text(rendered, limit)


def _remove_prior_evidence_messages(messages: list[dict]) -> None:
    messages[:] = [
        message for message in messages
        if not (
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and str(message.get("content")).startswith(_EVIDENCE_SUMMARY_PREFIX)
        )
    ]


def _summarize_evidence_state(
    *,
    question: str,
    user_content: Any | None,
    task: str | None,
    previous_state: dict[str, Any],
    pending_evidence: list[dict[str, Any]],
    timeout: float,
) -> tuple[dict[str, Any] | None, str, str | None]:
    if not pending_evidence:
        return None, "", None
    try:
        system = (
            "You maintain a concise evidence and constraint state for a web-research ReAct agent. "
            "Given the original prompt, optional image, previous state, and new tool observations, update the living state. "
            "Do not solve from memory or guess. Extract only constraints, candidate entities, verified facts, missing checks, and targeted next searches. "
            "Rejections are tentative: never permanently eliminate a candidate unless a cited observation directly contradicts it. "
            "If a browser observation contains source_warning or access_warning, do not treat that page as evidence for a new entity or as a contradiction; "
            "instead keep the prior candidate and suggest a same-entity targeted web_search for the missing answer field. "
            "Preserve prior high-confidence candidates unless a source-backed fact proves they fail a required constraint. "
            "Candidate confidence must be low if a required constraint is contradicted or if there is no evidence for the answer-specific field; "
            "confidence can be high only when the candidate and final answer span are both source-backed. "
            "Return strict JSON only with keys: summary, constraints, candidates, verified_facts, tentative_rejections, next_searches, pitfalls."
        )
        payload = {
            "task": task or "",
            "original_prompt": _clip_prompt_text(question, 2400),
            "previous_evidence_state": previous_state or {},
            "new_tool_observations": pending_evidence[-6:],
            "schema": {
                "constraints": [
                    {
                        "constraint": "required clue or answer-format requirement",
                        "status": "satisfied|missing|contradicted|unknown",
                        "evidence": "short source-backed note",
                    }
                ],
                "candidates": [
                    {
                        "name": "candidate entity/answer",
                        "support": "which constraints it satisfies",
                        "missing": "what still needs verification",
                        "conflicts": "contradictions if any",
                        "confidence": "low|medium|high",
                    }
                ],
                "next_searches": ["targeted query or page to open next"],
            },
            "rules": [
                "Keep the state compact and current; merge with previous state instead of appending duplicates.",
                "Prefer source-backed facts from browser/page observations over snippets.",
                "A redirect/source mismatch is not source-backed evidence for a candidate switch.",
                "Do not mark a candidate as primary/strongest if your own state says it fails a required constraint.",
                "Do not include final-answer wording unless it is a candidate that still needs verification.",
                "Do not mention gold answers, benchmark names, or datasets.",
            ],
        }
        user_text = json.dumps(payload, ensure_ascii=False)
        image_parts = _image_parts_for_summary(user_content)
        summary_user: str | list[dict[str, Any]]
        if image_parts:
            summary_user = [{"type": "text", "text": user_text}, *image_parts]
        else:
            summary_user = user_text
        response = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": summary_user}],
            tools=None,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=_env_int("SII_EVIDENCE_SUMMARY_MAX_TOKENS", 1000, 300, 2000),
            timeout=timeout,
        )
        raw_response = _response_text(response)
        parsed = _parse_json_object_from_text(raw_response)
        if not isinstance(parsed, dict):
            fallback = _normalize_evidence_state(
                {
                    "summary": "Evidence summary fallback: JSON parsing failed, so preserve prior state and compact the latest tool observations.",
                    "constraints": previous_state.get("constraints", []) if isinstance(previous_state, dict) else [],
                    "candidates": previous_state.get("candidates", []) if isinstance(previous_state, dict) else [],
                    "verified_facts": [
                        {
                            "tool": item.get("tool"),
                            "note": _clip_prompt_text(item.get("observation"), 220),
                        }
                        for item in pending_evidence[-4:]
                    ],
                    "tentative_rejections": previous_state.get("tentative_rejections", []) if isinstance(previous_state, dict) else [],
                    "next_searches": previous_state.get("next_searches", []) if isinstance(previous_state, dict) else [],
                    "pitfalls": previous_state.get("pitfalls", []) if isinstance(previous_state, dict) else [],
                },
                previous_state,
            )
            return fallback, _render_evidence_state(fallback), None
        state = _normalize_evidence_state(parsed, previous_state)
        rendered = _render_evidence_state(state)
        return state, rendered, None
    except Exception as exc:  # noqa: BLE001
        return None, "", f"{type(exc).__name__}: {_clip_prompt_text(exc, 240)}"


def _context_compact_enabled() -> bool:
    return _env_bool("SII_AGENT_CONTEXT_COMPACT", False)


def _render_compact_context(state: dict[str, Any], max_chars: int | None = None) -> str:
    limit = max_chars or _env_int("SII_CONTEXT_COMPACT_MAX_CHARS", 3500, 1200, 8000)
    return _clip_prompt_text(json.dumps(state or {}, ensure_ascii=False, indent=2), limit)


def _normalize_compact_context(parsed: dict[str, Any], previous: dict[str, Any], step: int) -> dict[str, Any]:
    state = dict(previous or {})
    for key, limit in (
        ("task_summary", 700),
        ("answer_target", 300),
        ("current_status", 500),
    ):
        if parsed.get(key) is not None:
            state[key] = _clip_prompt_text(parsed.get(key), limit)
    list_specs = {
        "constraints": (12, 260),
        "evidence": (14, 260),
        "candidates": (10, 300),
        "tool_history": (12, 220),
        "rejected_or_contradicted": (8, 240),
        "unresolved": (8, 220),
        "next_plan": (6, 220),
        "memory_guidance": (6, 220),
        "pitfalls": (6, 180),
    }
    for key, (limit, item_chars) in list_specs.items():
        normalized = _normalize_evidence_list(parsed.get(key), limit, item_chars)
        if normalized:
            state[key] = normalized
        else:
            state.setdefault(key, [])
    state["compacted_at_step"] = step
    return state


def _compact_trajectory_for_context(trajectory: list[dict], max_events: int = 36) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for event in trajectory[-max_events:]:
        if not isinstance(event, dict):
            continue
        role = str(event.get("role") or "")
        if role == "assistant":
            calls = []
            for tc in event.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                calls.append(
                    {
                        "name": str(fn.get("name") or ""),
                        "arguments": _clip_prompt_text(fn.get("arguments"), 500),
                    }
                )
            events.append(
                {
                    "role": "assistant",
                    "content": _clip_prompt_text(event.get("content"), 500),
                    "tool_calls": calls[:4],
                }
            )
        elif role == "tool":
            events.append(
                {
                    "role": "tool",
                    "name": str(event.get("name") or ""),
                    "args": _clip_prompt_text(event.get("args"), 500),
                    "content": _clip_prompt_text(event.get("content"), 1400),
                }
            )
        elif role == "system":
            events.append(
                {
                    "role": "system",
                    "name": str(event.get("name") or "system"),
                    "content": _clip_prompt_text(event.get("content"), 1200),
                }
            )
    return events


def _is_user_prefix_message(message: dict, prefix: str) -> bool:
    return (
        message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and str(message.get("content")).startswith(prefix)
    )


def _is_generated_context_message(message: dict) -> bool:
    return _is_user_prefix_message(message, _EVIDENCE_SUMMARY_PREFIX) or _is_user_prefix_message(
        message, _COMPACT_CONTEXT_PREFIX
    )


def _is_budget_or_control_message(message: dict) -> bool:
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        return False
    content = str(message.get("content") or "")
    prefixes = (
        "You have used the evidence tool budget",
        "You have already used ",
        "You have used ",
        "This is the final step.",
        "This is the final training step.",
        "The tool budget is exhausted.",
        "You did not call a tool.",
    )
    return content.startswith(prefixes)


def _latest_user_prefix_message(messages: list[dict], prefix: str) -> dict | None:
    for message in reversed(messages):
        if _is_user_prefix_message(message, prefix):
            return dict(message)
    return None


def _latest_budget_messages(messages: list[dict], limit: int = 3) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for message in reversed(messages):
        if not _is_budget_or_control_message(message):
            continue
        content = str(message.get("content") or "")
        if content in seen:
            continue
        out.append(dict(message))
        seen.add(content)
        if len(out) >= limit:
            break
    out.reverse()
    return out


def _safe_recent_message_suffix(messages: list[dict], keep_messages: int) -> list[dict]:
    if keep_messages <= 0 or not messages:
        return []
    start = max(0, len(messages) - keep_messages)
    while start > 0 and messages[start].get("role") == "tool":
        start -= 1
    if messages[start].get("role") == "tool":
        return []
    return [dict(message) for message in messages[start:]]


def _rebuild_messages_after_compaction(
    messages: list[dict],
    compact_note: dict,
    *,
    keep_recent_messages: int,
) -> None:
    if len(messages) < 2:
        messages.append(compact_note)
        return
    system_message = dict(messages[0])
    original_user_message = dict(messages[1])
    latest_evidence = _latest_user_prefix_message(messages, _EVIDENCE_SUMMARY_PREFIX)
    budget_messages = _latest_budget_messages(messages)
    filtered_tail = [
        dict(message) for message in messages[2:]
        if not _is_generated_context_message(message) and not _is_budget_or_control_message(message)
    ]
    recent = _safe_recent_message_suffix(filtered_tail, keep_recent_messages)
    rebuilt = [system_message, original_user_message, compact_note]
    if latest_evidence:
        rebuilt.append(latest_evidence)
    rebuilt.extend(budget_messages)
    rebuilt.extend(recent)
    messages[:] = rebuilt


def _summarize_compact_context(
    *,
    question: str,
    user_content: Any | None,
    task: str | None,
    previous_compact: dict[str, Any],
    evidence_state: dict[str, Any],
    trajectory: list[dict],
    step: int,
    timeout: float,
) -> tuple[dict[str, Any] | None, str, str | None]:
    try:
        system = (
            "You compact a ReAct agent's earlier context into structured state. "
            "Preserve all information needed to continue solving: original answer target, constraints, source-backed evidence, "
            "candidate status, useful tool history, memory guidance, rejected/contradicted candidates, unresolved checks, and next plan. "
            "Do not invent facts or final answers. Rejections are tentative unless directly contradicted by evidence. "
            "If recent trace shows a redirect/source mismatch or blocked page, preserve the prior candidate and mark the page as unusable; "
            "do not switch entities based only on that page. "
            "Do not promote candidates that fail required constraints; keep them as rejected_or_contradicted or low-confidence only. "
            "Return strict JSON only with keys: task_summary, answer_target, current_status, constraints, evidence, candidates, "
            "tool_history, rejected_or_contradicted, unresolved, next_plan, memory_guidance, pitfalls."
        )
        payload = {
            "task": task or "",
            "step": step,
            "original_prompt": _clip_prompt_text(question, 2400),
            "previous_compact_context": previous_compact or {},
            "latest_evidence_state": evidence_state or {},
            "recent_trace": _compact_trajectory_for_context(
                trajectory,
                _env_int("SII_CONTEXT_COMPACT_TRACE_EVENTS", 36, 8, 80),
            ),
            "rules": [
                "Keep only durable facts and decisions needed for future steps.",
                "For each candidate, note supported constraints, missing checks, and conflicts.",
                "Preserve prior high-confidence candidates unless source-backed evidence proves they fail a required constraint.",
                "Keep next_plan as targeted searches or pages to open, not broad instructions.",
                "Do not mention gold answers, benchmark names, or datasets.",
            ],
        }
        user_text = json.dumps(payload, ensure_ascii=False)
        image_parts = _image_parts_for_summary(user_content)
        compact_user: str | list[dict[str, Any]]
        if image_parts:
            compact_user = [{"type": "text", "text": user_text}, *image_parts]
        else:
            compact_user = user_text
        response = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": compact_user}],
            tools=None,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=_env_int("SII_CONTEXT_COMPACT_MAX_TOKENS", 1400, 500, 3000),
            timeout=timeout,
        )
        raw_response = _response_text(response)
        parsed = _parse_json_object_from_text(raw_response)
        if not isinstance(parsed, dict):
            fallback = _normalize_compact_context(
                {
                    "task_summary": "Compact fallback from existing evidence state and recent trace.",
                    "answer_target": _clip_prompt_text(question, 300),
                    "current_status": _clip_prompt_text((evidence_state or {}).get("summary"), 500),
                    "constraints": (evidence_state or {}).get("constraints") or [],
                    "evidence": (evidence_state or {}).get("verified_facts") or [],
                    "candidates": (evidence_state or {}).get("candidates") or [],
                    "tool_history": [
                        {
                            "tool": item.get("name") or item.get("role"),
                            "note": _clip_prompt_text(item.get("content"), 180),
                        }
                        for item in _compact_trajectory_for_context(trajectory, 12)
                        if item.get("role") in {"tool", "system"}
                    ],
                    "rejected_or_contradicted": (evidence_state or {}).get("tentative_rejections") or [],
                    "unresolved": [],
                    "next_plan": (evidence_state or {}).get("next_searches") or [],
                    "memory_guidance": [],
                    "pitfalls": (evidence_state or {}).get("pitfalls") or [],
                },
                previous_compact,
                step,
            )
            return fallback, _render_compact_context(fallback), None
        compact = _normalize_compact_context(parsed, previous_compact, step)
        rendered = _render_compact_context(compact)
        return compact, rendered, None
    except Exception as exc:  # noqa: BLE001
        return None, "", f"{type(exc).__name__}: {_clip_prompt_text(exc, 240)}"


def run_react(
    question: str,
    cfg: HarnessConfig | None = None,
    extra_system: str | None = None,
    expected: str | None = None,
    task: str | None = None,
    user_content: Any | None = None,
) -> HarnessResult:
    cfg = cfg or HarnessConfig()
    guard = StepGuard(cfg)
    res = HarnessResult()
    short_memory = ShortTermMemory(question, max_chars=cfg.short_memory_max_chars) if cfg.use_short_memory else None

    allowed_tools = cfg.allowed_tools or _profile_tools()
    allowed_tools = tuple(name for name in allowed_tools if name != "verify")
    if _runtime_mode() == "test":
        allowed_tools = tuple(name for name in allowed_tools if name not in TRAIN_ONLY_TOOLS)
    internal_verify_enabled = _internal_gold_verify_enabled(expected)
    verify_token = None
    if internal_verify_enabled:
        from tools.verify import set_verify_context

        verify_token = set_verify_context(question=question, expected=expected, task=task)

    def reset_verify() -> None:
        nonlocal verify_token
        if verify_token is not None:
            from tools.verify import reset_verify_context

            reset_verify_context(verify_token)
            verify_token = None

    sys = _system_prompt(allowed_tools, extra_system)
    messages: list[dict] = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user_content if user_content is not None else question},
    ]
    set_tool_context(question=question, user_content=user_content if user_content is not None else question, task=task)
    if _memory_auto_prefetch_enabled(allowed_tools):
        _auto_prefetch_memory(res, messages, question, user_content, task)
    has_verified = not internal_verify_enabled
    budget_messages_sent: set[str] = set()
    evidence_enabled = _evidence_summary_enabled(allowed_tools)
    context_compact_enabled = _context_compact_enabled()
    evidence_state: dict[str, Any] = {}
    compact_context: dict[str, Any] = {}
    pending_evidence: list[dict[str, Any]] = []
    web_searches_since_summary = 0
    compact_failures = 0

    def append_budget_message_once(key: str, content: str) -> None:
        if key in budget_messages_sent:
            return
        budget_messages_sent.add(key)
        messages.append({"role": "user", "content": content})
        res.trajectory.append({"role": "system", "name": "budget_notice", "content": content})

    def record_verify_result(tool_result: str) -> dict | None:
        nonlocal has_verified
        parsed = _safe_json(tool_result)
        if isinstance(parsed, dict) and isinstance(parsed.get("correct"), bool):
            if parsed["correct"]:
                has_verified = True
        return parsed

    def internal_verify(answer: str, rationale: str = "") -> tuple[str, dict | None]:
        from tools.verify import verify

        verify_result = verify(answer, rationale)
        parsed_verify = record_verify_result(verify_result)
        verify_row = dict(parsed_verify) if isinstance(parsed_verify, dict) else {"raw": verify_result}
        verify_row.setdefault("answer", answer)
        verify_row.setdefault("rationale", rationale)
        res.internal_verify_results.append(verify_row)
        res.tool_call_counts["internal_verify"] = res.tool_call_counts.get("internal_verify", 0) + 1
        res.trajectory.append(
            {
                "role": "system",
                "name": "internal_verify",
                "args": {"answer": answer, "rationale": rationale},
                "content": verify_result,
            }
        )
        return verify_result, parsed_verify

    for step in range(cfg.max_steps):
        active_tools = allowed_tools
        turn_evidence_summary_due = False
        pending_train_actions = internal_verify_enabled and not has_verified
        research_calls = _tool_count(res.tool_call_counts, _RESEARCH_TOOLS)
        web_search_calls = int(res.tool_call_counts.get("web_search") or 0)
        if (
            not pending_train_actions
            and "browsecomp_search" in allowed_tools
            and int(res.tool_call_counts.get("browsecomp_search") or 0) <= 0
            and _env_bool("SII_BROWSECOMP_FORCE_FIRST_SEARCH", False)
        ):
            active_tools = ("browsecomp_search",)
            append_budget_message_once(
                "browsecomp_first",
                (
                    "This text-only row has access to the official local BrowseComp BM25 index. "
                    "First call `browsecomp_search` with distinctive phrases/constraints from the question. "
                    "Use live web_search only after checking the local corpus."
                ),
            )
        if not pending_train_actions:
            if cfg.max_research_tool_calls > 0 and research_calls >= cfg.max_research_tool_calls:
                active_tools = ("final_answer",)
                append_budget_message_once(
                    "research_cap",
                    (
                        f"You have used the evidence tool budget ({research_calls}/{cfg.max_research_tool_calls}). "
                        "Stop searching and call `final_answer` now with the best concise answer supported by the "
                        "evidence already gathered. The answer field must be only the concise answer span, not an explanation."
                    ),
                )
            else:
                if cfg.max_web_search_calls > 0 and web_search_calls >= cfg.max_web_search_calls:
                    if "web_search" in active_tools:
                        active_tools = tuple(name for name in active_tools if name != "web_search")
                    append_budget_message_once(
                        "web_search_cap",
                        (
                            f"You have already used {web_search_calls} web_search calls. Stop broad web searching; "
                            "use a targeted page/wiki/image check only if essential, otherwise call `final_answer` with only a concise answer span."
                        ),
                    )
                if cfg.synthesize_after_tool_calls > 0 and research_calls >= cfg.synthesize_after_tool_calls:
                    append_budget_message_once(
                        "synthesize_notice",
                        (
                            f"You have used {research_calls} evidence tool calls. Converge now: identify the strongest "
                            "candidate, run at most one targeted verification if needed, then call `final_answer` with only the answer span."
                        ),
                    )
        if step == cfg.max_steps - 1 and not pending_train_actions:
            active_tools = ("final_answer",)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "This is the final step. Submit your best concise answer now with `final_answer`. "
                        "The answer field must be only a name/title/date/number/short phrase; do not include uncertainty or explanation."
                    ),
                }
            )
        elif step == cfg.max_steps - 1 and pending_train_actions:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "This is the final training step. If no candidate has been checked yet, call `final_answer` "
                        "with your current best answer so the training harness can verify it internally. If a prior "
                        "answer was rejected, use that feedback to revise the answer and call `final_answer` again."
                    ),
                }
            )
        specs = tool_specs(active_tools)
        time_left = guard.time_left()
        if time_left <= 0:
            res.stop_reason = "timeout"
            break
        if time_left < cfg.min_llm_call_seconds:
            res.stop_reason = "timeout"
            break
        try:
            tool_choice: str | dict = "auto"
            if active_tools == ("final_answer",):
                tool_choice = _forced_final_tool_choice()
            call_messages = _messages_with_short_memory(messages, short_memory, active_tools)
            resp = chat(
                call_messages,
                tools=specs,
                max_tokens=cfg.max_llm_tokens,
                timeout=min(cfg.max_llm_call_seconds, time_left),
                tool_choice=tool_choice,
            )
        except Exception as e:  # noqa: BLE001
            res.stop_reason = f"error: {type(e).__name__}: {e}"
            break

        choice = resp.choices[0]
        finish_reason = str(getattr(choice, "finish_reason", "") or "unknown")
        res.finish_reasons[finish_reason] = res.finish_reasons.get(finish_reason, 0) + 1
        msg = choice.message
        visible_content, inline_reasoning = _split_inline_reasoning(msg.content or "")
        parsed_content_tool_calls = [] if msg.tool_calls else _parse_content_tool_calls(visible_content)
        effective_tool_calls = list(msg.tool_calls or []) or parsed_content_tool_calls
        dumped_tool_calls = [_dump_tool_call(tc, active_tools) for tc in effective_tool_calls]
        tool_calls_to_run = effective_tool_calls
        skipped_parallel_tool_calls: list[Any] = []
        if cfg.max_parallel_tool_calls > 0 and len(effective_tool_calls) > cfg.max_parallel_tool_calls:
            tool_calls_to_run = effective_tool_calls[: cfg.max_parallel_tool_calls]
            skipped_parallel_tool_calls = effective_tool_calls[cfg.max_parallel_tool_calls :]
        final_tool_calls = [
            tc for tc in effective_tool_calls
            if _canonical_tool_name(str(tc.function.name or "").strip(), active_tools) == "final_answer"
        ]
        if final_tool_calls:
            tool_calls_to_run = [final_tool_calls[0]]
            skipped_parallel_tool_calls = []
        assistant_content = "" if parsed_content_tool_calls else visible_content
        reasoning_content = "\n\n".join(
            part for part in (_message_reasoning(msg).strip(), inline_reasoning) if part
        )
        assistant_message = {"role": "assistant", "content": assistant_content}
        if effective_tool_calls:
            assistant_message["tool_calls"] = dumped_tool_calls
        messages.append(assistant_message)
        res.steps = step + 1
        assistant_event = {"role": "assistant", "content": visible_content, "tool_calls": dumped_tool_calls}
        if reasoning_content:
            assistant_event["reasoning_content"] = reasoning_content
        res.trajectory.append(assistant_event)

        if not effective_tool_calls:
            content = visible_content.strip()
            if active_tools == ("final_answer",) and content and not _looks_invalid_final_answer(content):
                if internal_verify_enabled and not has_verified:
                    verify_result, parsed_verify = internal_verify(content, "text finalization on forced final step")
                    if not (isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True):
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "INTERNAL_GOLD_VERIFY_RESULT_BEFORE_FINAL_ANSWER:\n"
                                    f"{verify_result}\n\n"
                                    "The candidate was not accepted. Reflect on the failure, then call `final_answer` "
                                    "with a corrected answer. Do not create lesson/skill memory yet; memory is written "
                                    "only if the corrected answer verifies as correct."
                                ),
                            }
                        )
                        continue
                res.final_answer = content
                res.rationale = "text finalization on forced final step"
                res.stop_reason = "final"
                _append_tool_event(
                    res,
                    "final_answer",
                    {"answer": content, "rationale": res.rationale},
                    "OK: accepted text finalization on forced final step.",
                )
                if short_memory is not None:
                    res.short_memory_stats = short_memory.stats()
                res.elapsed = guard.elapsed()
                reset_verify()
                return res
            if active_tools == ("final_answer",) and content:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous response was not a concise answer span. Call `final_answer` with only the "
                            "best candidate name/title/date/number; do not explain uncertainty in the answer field."
                        ),
                    }
                )
                continue
            if guard.note_no_tool():
                res.stop_reason = "loop:no_tool"
                break
            if short_memory is not None:
                short_memory.observe_no_tool(content)
            messages.append({"role": "user", "content": "You did not call a tool. Either call a tool or call `final_answer`."})
            continue
        guard.reset_no_tool()

        for tc in tool_calls_to_run:
            original_name = str(tc.function.name or "").strip()
            name = _canonical_tool_name(original_name, active_tools)
            args = _safe_json(tc.function.arguments or "{}")

            if name not in active_tools:
                tool_result = (
                    f"ERROR: tool '{original_name or name}' is not allowed in this run. "
                    f"Allowed tools: {list(active_tools)}"
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
                _append_tool_event(res, name, {}, tool_result, original_name=original_name)
                continue

            if args is None:
                tool_result = (
                    "ERROR: invalid JSON arguments. Retry with valid JSON for this tool; "
                    "if submitting the final answer, keep it concise."
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result,
                    }
                )
                _append_tool_event(res, name, {}, tool_result, original_name=original_name)
                continue
            args = _with_browser_extract_query(name, args, question)
            set_tool_context(
                question=question,
                user_content=user_content if user_content is not None else question,
                task=task,
                trajectory=list(res.trajectory),
                step=res.steps,
            )
            res.tool_calls += 1
            res.tool_call_counts[name] = res.tool_call_counts.get(name, 0) + 1

            if name == "final_answer":
                if internal_verify_enabled and not has_verified:
                    answer = str(args.get("answer", "")).strip()
                    if not answer:
                        tool_result = (
                            "ERROR: final_answer requires a concise answer span only (name/title/date/number), not "
                            "an explanation or 'not found' narrative. Retry with the best concise candidate answer."
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tool_result,
                            }
                        )
                        _append_tool_event(res, name, args, tool_result, original_name=original_name)
                        continue
                    rationale = str(args.get("rationale") or "")
                    verify_result, parsed_verify = internal_verify(answer, rationale)
                    if isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True:
                        if not _apply_final_tool_call(res, tc):
                            tool_result = (
                                "ERROR: final_answer requires a concise answer span only (name/title/date/number), not "
                                "an explanation or 'not found' narrative. Retry with the best concise candidate answer."
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": tool_result,
                                }
                            )
                            _append_tool_event(res, name, args, tool_result, original_name=original_name)
                            continue
                        _append_tool_event(res, name, args, "OK: final answer accepted.", original_name=original_name)
                        res.elapsed = guard.elapsed()
                        if short_memory is not None:
                            res.short_memory_stats = short_memory.stats()
                        reset_verify()
                        return res
                    tool_result = (
                        "INTERNAL_GOLD_VERIFY_RESULT_BEFORE_FINAL_ANSWER:\n"
                        f"{verify_result}\n\n"
                        "The candidate was not accepted. If correct=false, reflect on the failure, then call "
                        "`final_answer` with a corrected answer. Do not create lesson/skill memory yet; memory is "
                        "written only if the corrected answer verifies as correct."
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )
                    _append_tool_event(res, name, args, tool_result, original_name=original_name)
                    continue
                if not _apply_final_tool_call(res, tc):
                    tool_result = (
                        "ERROR: final_answer requires a concise answer span only (name/title/date/number), not "
                        "an explanation or 'not found' narrative. Retry with the best concise candidate answer."
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_result,
                        }
                    )
                    _append_tool_event(res, name, args, tool_result, original_name=original_name)
                    continue
                _append_tool_event(res, name, args, "OK: final answer accepted.", original_name=original_name)
                res.elapsed = guard.elapsed()
                if short_memory is not None:
                    res.short_memory_stats = short_memory.stats()
                reset_verify()
                return res

            if guard.note_tool(name, json.dumps(args, sort_keys=True)):
                tool_result = (
                    f"NOTICE: You already called {name} with these args multiple times. "
                    f"Try a different query, a different tool, or call final_answer."
                )
            else:
                tool_result = dispatch(name, args)
                if short_memory is not None:
                    short_memory.observe_tool(name, args, tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )
            _append_tool_event(res, name, args, tool_result, original_name=original_name)
            if (
                evidence_enabled
                and name in _RESEARCH_TOOLS
                and not str(tool_result).startswith(("ERROR:", "NOTICE:"))
            ):
                pending_evidence.append(_compact_tool_observation(name, args, tool_result))
                if len(pending_evidence) > 8:
                    pending_evidence = pending_evidence[-8:]
                if name == "web_search":
                    web_searches_since_summary += 1
                if name in _EVIDENCE_SUMMARY_IMMEDIATE_TOOLS:
                    turn_evidence_summary_due = True

        for tc in skipped_parallel_tool_calls:
            original_name = str(tc.function.name or "").strip()
            name = _canonical_tool_name(original_name, active_tools)
            tool_result = (
                "NOTICE: extra parallel tool call skipped because this turn exceeded the configured tool-call cap; "
                "read the executed tool result, then call one next tool or `final_answer`."
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )
            skipped_args = _safe_json(tc.function.arguments or "{}") or {}
            _append_tool_event(res, name, skipped_args, tool_result, original_name=original_name)

        if evidence_enabled and pending_evidence and not res.final_answer:
            web_every = _env_int("SII_EVIDENCE_SUMMARY_WEB_EVERY", 3, 1, 12)
            should_summarize = turn_evidence_summary_due or web_searches_since_summary >= web_every
            time_left = guard.time_left()
            min_time_left = _env_int("SII_EVIDENCE_SUMMARY_MIN_TIME_LEFT", 70, 20, 180)
            if should_summarize and time_left >= min_time_left:
                summary_timeout = min(
                    float(_env_int("SII_EVIDENCE_SUMMARY_TIMEOUT", 30, 5, 90)),
                    max(1.0, time_left - cfg.min_llm_call_seconds),
                )
                updated_state, rendered_state, summary_error = _summarize_evidence_state(
                    question=question,
                    user_content=user_content if user_content is not None else question,
                    task=task,
                    previous_state=evidence_state,
                    pending_evidence=list(pending_evidence),
                    timeout=summary_timeout,
                )
                if updated_state is not None and rendered_state:
                    summarized_count = len(pending_evidence)
                    evidence_state = updated_state
                    res.evidence_state = dict(evidence_state)
                    pending_evidence.clear()
                    web_searches_since_summary = 0
                    _remove_prior_evidence_messages(messages)
                    note = (
                        f"{_EVIDENCE_SUMMARY_PREFIX}\n"
                        "Use this as the current living constraint table. It is a synthesis of gathered evidence, "
                        "not independent evidence; verify unresolved constraints with tools.\n"
                        f"{rendered_state}"
                    )
                    messages.append({"role": "user", "content": note})
                    res.trajectory.append(
                        {
                            "role": "system",
                            "name": "evidence_summary",
                            "args": {"pending_evidence_count": summarized_count, "web_every": web_every},
                            "content": rendered_state,
                        }
                    )
                elif summary_error:
                    res.trajectory.append(
                        {
                            "role": "system",
                            "name": "evidence_summary_error",
                            "content": summary_error,
                        }
                    )

        if guard.time_left() <= 0:
            res.stop_reason = "timeout"
            break

        if context_compact_enabled and not res.final_answer:
            compact_every = _env_int("SII_CONTEXT_COMPACT_EVERY", 6, 2, 24)
            should_compact = res.steps >= compact_every and res.steps % compact_every == 0
            if should_compact and compact_failures < _env_int("SII_CONTEXT_COMPACT_MAX_FAILURES", 2, 1, 5):
                time_left = guard.time_left()
                min_time_left = _env_int("SII_CONTEXT_COMPACT_MIN_TIME_LEFT", 80, 20, 240)
                if time_left >= min_time_left:
                    compact_timeout = min(
                        float(_env_int("SII_CONTEXT_COMPACT_TIMEOUT", 30, 5, 120)),
                        max(1.0, time_left - cfg.min_llm_call_seconds),
                    )
                    updated_compact, rendered_compact, compact_error = _summarize_compact_context(
                        question=question,
                        user_content=user_content if user_content is not None else question,
                        task=task,
                        previous_compact=compact_context,
                        evidence_state=evidence_state,
                        trajectory=list(res.trajectory),
                        step=res.steps,
                        timeout=compact_timeout,
                    )
                    if updated_compact is not None and rendered_compact:
                        compact_context = updated_compact
                        compact_failures = 0
                        res.compact_context = dict(compact_context)
                        compact_note = {
                            "role": "user",
                            "content": (
                                f"{_COMPACT_CONTEXT_PREFIX}\n"
                                "This is the compressed history of earlier steps. Use it as persistent context; "
                                "recent raw tool messages that follow may contain newer details.\n"
                                f"{rendered_compact}"
                            ),
                        }
                        _rebuild_messages_after_compaction(
                            messages,
                            compact_note,
                            keep_recent_messages=_env_int("SII_CONTEXT_COMPACT_KEEP_RECENT_MESSAGES", 12, 4, 40),
                        )
                        res.trajectory.append(
                            {
                                "role": "system",
                                "name": "context_compact",
                                "args": {
                                    "step": res.steps,
                                    "every": compact_every,
                                    "messages_after": len(messages),
                                },
                                "content": rendered_compact,
                            }
                        )
                    elif compact_error:
                        compact_failures += 1
                        res.trajectory.append(
                            {
                                "role": "system",
                                "name": "context_compact_error",
                                "args": {"step": res.steps, "failures": compact_failures},
                                "content": compact_error,
                            }
                        )

    if not res.final_answer and cfg.finalize_on_stop and guard.time_left() >= cfg.min_llm_call_seconds:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The tool budget is exhausted. Based only on the gathered evidence, "
                    "call `final_answer` now with the best concise answer. Return only the answer phrase/name "
                    "and optional citations/docids; do not add analysis or uncertainty caveats."
                ),
            }
        )
        try:
            call_messages = _messages_with_short_memory(messages, short_memory, ("final_answer",))
            resp = chat(
                call_messages,
                tools=tool_specs(("final_answer",)),
                tool_choice=_forced_final_tool_choice(),
                max_tokens=cfg.max_llm_tokens,
                timeout=min(cfg.max_llm_call_seconds, guard.time_left()),
            )
            choice = resp.choices[0]
            finish_reason = str(getattr(choice, "finish_reason", "") or "unknown")
            res.finish_reasons[f"forced_{finish_reason}"] = (
                res.finish_reasons.get(f"forced_{finish_reason}", 0) + 1
            )
            msg = choice.message
            visible_content, inline_reasoning = _split_inline_reasoning(msg.content or "")
            parsed_content_tool_calls = [] if msg.tool_calls else _parse_content_tool_calls(visible_content)
            effective_tool_calls = list(msg.tool_calls or []) or parsed_content_tool_calls
            dumped_tool_calls = [_dump_tool_call(tc, ("final_answer",)) for tc in effective_tool_calls]
            reasoning_content = "\n\n".join(
                part for part in (_message_reasoning(msg).strip(), inline_reasoning) if part
            )
            assistant_event = {"role": "assistant", "content": visible_content, "tool_calls": dumped_tool_calls}
            if reasoning_content:
                assistant_event["reasoning_content"] = reasoning_content
            res.trajectory.append(assistant_event)
            for tc in effective_tool_calls:
                name = _canonical_tool_name(str(tc.function.name or "").strip(), ("final_answer",))
                if name != "final_answer":
                    continue
                args = _safe_json(tc.function.arguments or "{}")
                if args is None:
                    continue
                answer, _ = _split_inline_reasoning(str(args.get("answer", "")))
                answer = answer.strip()
                if internal_verify_enabled and not has_verified and answer:
                    verify_result, parsed_verify = internal_verify(answer, str(args.get("rationale") or ""))
                    if not (isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True):
                        # No interaction budget remains here. Keep the rejected final answer for scoring;
                        # runner-level gold reflection will force lesson/skill persistence.
                        res.final_answer = answer
                        rationale, _ = _split_inline_reasoning(str(args.get("rationale") or ""))
                        res.rationale = rationale
                        res.stop_reason = "final"
                        res.tool_calls += 1
                        res.tool_call_counts["final_answer"] = res.tool_call_counts.get("final_answer", 0) + 1
                        _append_tool_event(
                            res,
                            "final_answer",
                            args,
                            "OK: rejected by internal verifier but kept after forced finalization.",
                            original_name=tc.function.name,
                        )
                        break
                if _apply_final_tool_call(res, tc):
                    res.tool_calls += 1
                    res.tool_call_counts["final_answer"] = res.tool_call_counts.get("final_answer", 0) + 1
                    _append_tool_event(
                        res,
                        "final_answer",
                        args,
                        "OK: final answer accepted.",
                        original_name=tc.function.name,
                    )
                    break
            content = visible_content.strip()
            if content and not res.final_answer:
                if internal_verify_enabled and not has_verified:
                    verify_result, parsed_verify = internal_verify(content, "forced finalization after tool budget")
                    if not (isinstance(parsed_verify, dict) and parsed_verify.get("correct") is True):
                        # No interaction budget remains here. Keep the rejected final answer for scoring;
                        # runner-level gold reflection will force lesson/skill persistence.
                        res.final_answer = content
                        res.rationale = "forced finalization after tool budget"
                        res.stop_reason = "final"
                res.final_answer = content
                res.rationale = "forced finalization after tool budget"
                res.stop_reason = "final"
                _append_tool_event(
                    res,
                    "final_answer",
                    {"answer": content, "rationale": res.rationale},
                    "OK: accepted forced text finalization after tool budget.",
                )
        except Exception as e:  # noqa: BLE001
            res.stop_reason = f"error: {type(e).__name__}: {e}"

    if not res.stop_reason:
        res.stop_reason = "max_steps"
    if short_memory is not None:
        res.short_memory_stats = short_memory.stats()
    res.elapsed = guard.elapsed()
    reset_verify()
    return res
