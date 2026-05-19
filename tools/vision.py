"""Vision/OCR tools backed by the configured OpenAI-compatible VLM endpoint."""
from __future__ import annotations

import base64
import json
import ipaddress
import mimetypes
import os
import re
import socket
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from dotenv import load_dotenv

from .registry import register

_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_ALLOWED_MIME_PREFIX = "image/"
_MAX_REDIRECTS = 5
_DEFAULT_PROMPT = (
    "Extract all readable text from the image, list concrete visual clues, and if an "
    "entity must be identified give multiple plausible candidates with uncertainty. "
    "Be concise and do not invent text."
)
_VISUAL_WEB_SEARCH_PROMPT = """You are doing the visual analysis stage for a visual factual QA task.

Question:
{question}

Analyze the image without anchoring on a single guess. Return STRICT JSON with this schema:
{{
  "ocr_text": ["literal visible text, if any"],
  "visual_clues": ["concrete observable clues, not guesses"],
  "answer_type": "person/place/year/object/style/other",
  "candidate_entities": [
    {{"name": "candidate entity", "confidence": 0.0, "why": "visual reason"}}
  ],
  "search_queries": ["queries that use OCR or visual clues plus the question predicate"]
}}

Rules:
- Include 2-5 candidate_entities when identity is uncertain.
- Include at least one search query based on OCR/visible clues rather than a guessed entity.
- If there is no reliable entity, leave candidate_entities empty and rely on OCR/visual clue queries.
- Do not answer the final question yet.
"""
_VISUAL_CANDIDATE_RETRY_PROMPT = """The previous visual pass did not produce enough named candidates.

Question:
{question}

Look at the image again and focus only on producing searchable exact candidates. If the image
shows a person, building, temple, painting, map, book, vehicle, logo, or place, list plausible
proper names. If exact names are impossible, list distinctive search phrases grounded in OCR or
visible clues.

Return STRICT JSON:
{{
  "candidate_entities": [
    {{"name": "possible exact entity or proper noun", "confidence": 0.0, "why": "visible clue"}}
  ],
  "search_queries": ["short query combining the visual clue/entity with the question predicate"]
}}
"""
_IMAGE_TO_SEARCH_QUERIES_PROMPT = """You generate web_search-ready text queries from an image.

Question or search goal:
{question}

Analyze the image and return STRICT JSON:
{{
  "ocr_text": ["literal visible text, if any"],
  "visual_clues": ["specific visible clues useful for search"],
  "candidate_entities": [
    {{"name": "possible exact person/place/object/work/logo/entity", "confidence": 0.0, "why": "visible clue"}}
  ],
  "search_queries": ["concise search query grounded in OCR/visual clues"]
}}

Rules:
- Generate 2-{max_queries} diverse, concise queries suitable for the `web_search` tool.
- Prefer OCR text, proper nouns, logos, signs, landmarks, distinctive objects, dates, and visual style clues.
- If an exact entity is uncertain, include alternate queries rather than one overconfident guess.
- Include the question/search goal terms when they help retrieve the needed fact.
- Do not answer the final question; only produce search queries and supporting visual clues.
- Return JSON only, without Markdown.
"""


def _env_model() -> str:
    load_dotenv()
    if os.getenv("VISION_BACKEND", os.getenv("LLM_BACKEND", "")).lower() == "vllm":
        return (
            os.getenv("VISION_MODEL")
            or os.getenv("OPD_EXPERT_MODEL")
            or os.getenv("VLLM_MODEL")
            or "Qwen/Qwen3.5-9B"
        )
    return (
        os.getenv("VISION_MODEL")
        or os.getenv("OPD_EXPERT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        or "gpt-5.4"
    )


def _env_base_url() -> str:
    load_dotenv()
    base_url = os.getenv("VISION_BASE_URL")
    if base_url:
        return base_url
    if os.getenv("VISION_BACKEND", os.getenv("LLM_BACKEND", "")).lower() == "vllm":
        return os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("AZURE_OPENAI_BASE_URL")
    if base_url:
        return base_url
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if endpoint:
        return endpoint.rstrip("/") + "/openai/v1/"
    raise ValueError("OPENAI_BASE_URL or AZURE_OPENAI_BASE_URL is required for image_to_text")


def _env_api_key() -> str:
    load_dotenv()
    if os.getenv("VISION_BACKEND", os.getenv("LLM_BACKEND", "")).lower() == "vllm":
        return os.getenv("VISION_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY"
    api_key = os.getenv("VISION_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY or AZURE_OPENAI_API_KEY is required for image_to_text")
    return api_key


def _is_http_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


def _assert_public_http_url(source: str) -> None:
    parsed = urlparse(source)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("source must be an http(s) image URL")
    host = parsed.hostname
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve image URL host: {host}") from exc
    for address in {item[4][0] for item in addresses}:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError(f"refusing to fetch non-public image URL host: {host}")


def _guess_mime(source: str, content_type: str | None = None) -> str:
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    if mime.startswith(_ALLOWED_MIME_PREFIX):
        return mime
    guessed, _ = mimetypes.guess_type(source)
    if guessed and guessed.startswith(_ALLOWED_MIME_PREFIX):
        return guessed
    return "image/png"


def _safe_error(exc: Exception, limit: int = 1200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"data:image/[^,\\s'\"]+,[-A-Za-z0-9+/=_%]+", "data:image/[redacted]", text)
    text = re.sub(r"data:[^,\\s'\"]+,[-A-Za-z0-9+/=_%]+", "data:[redacted]", text)
    return _truncate(text, limit)


def _strip_inline_reasoning(content: str) -> str:
    return re.sub(r"<think\b[^>]*>.*?</think>", "", content or "", flags=re.I | re.S).strip()


def _load_image(source: str) -> tuple[str, bytes]:
    if _is_http_url(source):
        current_url = source
        for _ in range(_MAX_REDIRECTS + 1):
            _assert_public_http_url(current_url)
            response = httpx.get(current_url, timeout=30, follow_redirects=False)
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    break
                current_url = urljoin(current_url, location)
                continue
            break
        else:
            raise ValueError(f"too many redirects while fetching image URL; max {_MAX_REDIRECTS}")
        response.raise_for_status()
        data = response.content
        content_type = (response.headers.get("content-type") or "").split(";", 1)[0].strip().lower()
        if content_type and not content_type.startswith(_ALLOWED_MIME_PREFIX):
            raise ValueError(f"source returned non-image content-type: {content_type}")
        mime = _guess_mime(str(response.url), response.headers.get("content-type"))
    else:
        path = Path(source).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"image file not found: {path}")
        data = path.read_bytes()
        mime = _guess_mime(str(path))

    if not mime.startswith(_ALLOWED_MIME_PREFIX):
        raise ValueError(f"source is not an image MIME type: {mime}")
    if len(data) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image is too large: {len(data)} bytes; max {_MAX_IMAGE_BYTES}")
    return mime, data


def _call_vision(model: str, image_data_url: str, prompt: str, max_tokens: int) -> str:
    from openai import OpenAI

    base_url = _env_base_url()
    client = OpenAI(api_key=_env_api_key(), base_url=base_url, timeout=90)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
    ]
    payload = {
        "model": model,
        "messages": messages,
    }
    if os.getenv("VLLM_ENABLE_THINKING", "0").lower() not in {"1", "true", "yes"} and (
        "127.0.0.1" in base_url or "localhost" in base_url
    ):
        payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    try:
        response = client.chat.completions.create(
            **payload,
            max_completion_tokens=max_tokens,
        )
    except TypeError:
        response = client.chat.completions.create(
            **payload,
            max_tokens=max_tokens,
        )
    message = response.choices[0].message
    content = getattr(message, "content", None)
    if content:
        return _strip_inline_reasoning(str(content))
    for attr in ("reasoning", "reasoning_content"):
        value = getattr(message, attr, None)
        if value:
            return str(value).strip()
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        value = extra.get(key)
        if value:
            return str(value).strip()
    return ""


def _json_loads_soft(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    if "```" in stripped:
        parts = stripped.split("```")
        for part in parts:
            body = part.strip()
            if body.startswith("json"):
                body = body[4:].strip()
            if body.startswith("{") and body.endswith("}"):
                candidates.append(body)
    start = stripped.find("{")
    if start >= 0:
        depth = 0
        for idx, ch in enumerate(stripped[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(stripped[start:idx + 1])
                    break
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _jsonish_list_values(text: str, key: str, limit: int = 8) -> list[str]:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]\s*(?:,|\}})', text, re.DOTALL)
    if not match:
        return []
    values = re.findall(r'"([^"]+)"', match.group(1), flags=re.DOTALL)
    return [" ".join(value.split()) for value in values[:limit] if value.strip()]


def _analysis_from_jsonish(text: str) -> dict[str, Any] | None:
    names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    queries = _jsonish_list_values(text, "search_queries")
    ocr_text = _jsonish_list_values(text, "ocr_text")
    clues = _jsonish_list_values(text, "visual_clues", limit=5)
    if not (names or queries or ocr_text or clues):
        return None
    return {
        "ocr_text": ocr_text,
        "visual_clues": clues,
        "candidate_entities": [{"name": " ".join(name.split())} for name in names if name.strip()],
        "search_queries": queries,
        "raw_parse_note": "Recovered from non-strict JSON emitted by the VLM.",
    }


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [part.strip() for part in value.replace("；", ";").split(";") if part.strip()]
    return [value]


def _candidate_name(candidate: Any) -> str:
    if isinstance(candidate, dict):
        return str(candidate.get("name") or candidate.get("entity") or "").strip()
    return str(candidate).strip()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "\n...[truncated]"


def _truncate_search_result(text: str, max_chars: int = 1800) -> str:
    sections = re.split(r"\n\n(?=## )", text)
    if len(sections) <= 1:
        return _truncate(text, max_chars)
    per_section = max(500, max_chars // len(sections))
    compact = "\n\n".join(_truncate(section, per_section) for section in sections)
    return _truncate(compact, max_chars)


def _dedupe(items: list[str], limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = " ".join(str(item).split())
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def _queries_from_analysis(
    question: str,
    analysis: dict[str, Any] | None,
    raw_analysis: str,
    max_candidates: int,
    max_search_queries: int,
) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    candidate_names: list[str] = []
    if analysis:
        ocr_text = " ".join(str(item) for item in _listify(analysis.get("ocr_text")) if str(item).strip())
        if ocr_text:
            queries.append(f"{ocr_text} {question}")
        for query in _listify(analysis.get("search_queries")):
            queries.append(str(query))
        for candidate in _listify(analysis.get("candidate_entities")):
            name = _candidate_name(candidate)
            if name and name.lower() not in {"unknown", "none", "n/a", "null"}:
                candidate_names.append(name)
                queries.append(f"{name} {question}")
            if len(candidate_names) >= max_candidates:
                break
        clues = " ".join(str(item) for item in _listify(analysis.get("visual_clues"))[:3] if str(item).strip())
        if clues:
            queries.append(f"{clues} {question}")
    if not queries and raw_analysis:
        queries.append(f"{_truncate(raw_analysis, 300)} {question}")
    if not queries:
        queries.append(question)
    return _dedupe(queries, max_search_queries), _dedupe(candidate_names, max_candidates)


def _search_many(queries: list[str], k: int, time_budget_s: float) -> list[dict[str, str]]:
    from .search import web_search

    deadline = time.monotonic() + max(5.0, time_budget_s)
    order = {query: idx for idx, query in enumerate(queries)}
    pool = ThreadPoolExecutor(max_workers=min(4, max(1, len(queries))))
    futures = {pool.submit(web_search, query, k): query for query in queries}
    results: list[dict[str, str]] = []
    try:
        while futures and time.monotonic() < deadline:
            timeout = max(0.1, deadline - time.monotonic())
            done, _ = wait(futures, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done:
                break
            for future in done:
                query = futures.pop(future)
                try:
                    text = future.result()
                except Exception as exc:  # noqa: BLE001
                    text = f"ERROR: search failed: {type(exc).__name__}: {exc}"
                results.append({"query": query, "result": _truncate_search_result(text)})
        for future, query in list(futures.items()):
            future.cancel()
            results.append({"query": query, "result": "TIMEOUT: search skipped by visual_web_search time budget"})
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    results.sort(key=lambda item: order.get(item["query"], len(order)))
    return results


def _compact_json(data: Any, max_chars: int) -> str:
    return _truncate(json.dumps(data, ensure_ascii=False, indent=2), max_chars)


def _merge_analysis(base: dict[str, Any] | None, extra: dict[str, Any] | None) -> dict[str, Any] | None:
    if base is None:
        return extra
    if extra is None:
        return base
    merged = dict(base)
    for key in ("ocr_text", "visual_clues", "candidate_entities", "search_queries"):
        merged[key] = _listify(base.get(key)) + _listify(extra.get(key))
    if extra.get("raw_parse_note"):
        merged["raw_parse_note"] = extra["raw_parse_note"]
    return merged


@register(
    "image_to_text",
    "Extract text/OCR and describe useful visual evidence from an image URL or local image path using the configured VLM endpoint. "
    "Use for image-to-text search, screenshots, document images, charts, or visual clue questions.",
    {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "http(s) image URL or local image path",
            },
            "prompt": {
                "type": "string",
                "default": _DEFAULT_PROMPT,
                "description": "Specific OCR/vision instruction",
            },
            "model": {
                "type": "string",
                "default": "configured VLM",
                "description": "Vision-capable OpenAI-compatible model; defaults to OPD_EXPERT_MODEL/AZURE_OPENAI_DEPLOYMENT.",
            },
            "max_tokens": {"type": "integer", "default": 512, "minimum": 64, "maximum": 2048},
        },
        "required": ["source"],
    },
)
def image_to_text(
    source: str,
    prompt: str = _DEFAULT_PROMPT,
    model: str | None = None,
    max_tokens: int = 512,
) -> str:
    try:
        max_tokens = max(64, min(2048, int(max_tokens)))
        mime, data = _load_image(source)
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        text = _call_vision(model or _env_model(), data_url, prompt or _DEFAULT_PROMPT, max_tokens)
        return text or "(no text returned)"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: image_to_text failed: {_safe_error(exc)}"


@register(
    "image_to_search_queries",
    "Generate web_search-ready text queries from an image using the configured VLM. "
    "Use when you need OCR/visual clues turned into search query text before calling web_search; this tool does not search.",
    {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "http(s) image URL or local image path",
            },
            "question": {
                "type": "string",
                "default": "",
                "description": "Optional question or target fact to guide query generation",
            },
            "model": {
                "type": "string",
                "default": "configured VLM",
                "description": "Vision-capable OpenAI-compatible model; defaults to OPD_EXPERT_MODEL/AZURE_OPENAI_DEPLOYMENT.",
            },
            "max_queries": {"type": "integer", "default": 5, "minimum": 1, "maximum": 8},
            "max_tokens": {"type": "integer", "default": 768, "minimum": 128, "maximum": 2048},
        },
        "required": ["source"],
    },
)
def image_to_search_queries(
    source: str,
    question: str = "",
    model: str | None = None,
    max_queries: int = 5,
    max_tokens: int = 768,
) -> str:
    try:
        max_queries = max(1, min(8, int(max_queries)))
        max_tokens = max(128, min(2048, int(max_tokens)))
        mime, data = _load_image(source)
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        prompt = _IMAGE_TO_SEARCH_QUERIES_PROMPT.format(
            question=question.strip() or "(none; generate general identification/search queries)",
            max_queries=max_queries,
        )
        raw_analysis = _call_vision(model or _env_model(), data_url, prompt, max_tokens)
        analysis = _json_loads_soft(raw_analysis) or _analysis_from_jsonish(raw_analysis)
        queries, candidate_names = _queries_from_analysis(
            question.strip(),
            analysis,
            raw_analysis,
            max_candidates=max_queries,
            max_search_queries=max_queries,
        )
        return _compact_json(
            {
                "queries": queries,
                "candidate_entities": candidate_names,
                "vision_analysis": analysis if analysis is not None else {"raw": _truncate(raw_analysis, 1500)},
                "usage_note": "Call web_search with the strongest query; refine with another query if results are weak.",
            },
            4200,
        )
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: image_to_search_queries failed: {_safe_error(exc)}"


@register(
    "visual_web_search",
    "For image-based factual questions, generate visual/OCR search queries and collect web/wiki evidence. "
    "This tool does not produce or submit a final answer.",
    {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "http(s) image URL or local image path",
            },
            "question": {
                "type": "string",
                "description": "The exact user question to answer from the image and evidence",
            },
            "model": {
                "type": "string",
                "default": "configured VLM",
                "description": "Vision-capable OpenAI-compatible model; defaults to OPD_EXPERT_MODEL/AZURE_OPENAI_DEPLOYMENT.",
            },
            "max_candidates": {"type": "integer", "default": 4, "minimum": 1, "maximum": 6},
            "max_search_queries": {"type": "integer", "default": 6, "minimum": 2, "maximum": 8},
            "k": {"type": "integer", "default": 3, "minimum": 1, "maximum": 5},
            "time_budget_s": {"type": "number", "default": 55, "minimum": 10, "maximum": 120},
            "max_tokens": {"type": "integer", "default": 1536, "minimum": 256, "maximum": 4096},
        },
        "required": ["source", "question"],
    },
)
def visual_web_search(
    source: str,
    question: str,
    model: str | None = None,
    max_candidates: int = 4,
    max_search_queries: int = 6,
    k: int = 3,
    time_budget_s: float = 55,
    max_tokens: int = 1536,
) -> str:
    try:
        max_candidates = max(1, min(6, int(max_candidates)))
        max_search_queries = max(2, min(8, int(max_search_queries)))
        k = max(1, min(5, int(k)))
        time_budget_s = max(10.0, min(120.0, float(time_budget_s)))
        max_tokens = max(256, min(4096, int(max_tokens)))
        model_name = model or _env_model()
        total_start = time.monotonic()

        mime, data = _load_image(source)
        data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        analysis_prompt = _VISUAL_WEB_SEARCH_PROMPT.format(question=question)
        raw_analysis = _call_vision(model_name, data_url, analysis_prompt, max_tokens)
        analysis = _json_loads_soft(raw_analysis) or _analysis_from_jsonish(raw_analysis)
        if not [_candidate_name(item) for item in _listify((analysis or {}).get("candidate_entities"))]:
            retry_prompt = _VISUAL_CANDIDATE_RETRY_PROMPT.format(question=question)
            raw_retry = _call_vision(model_name, data_url, retry_prompt, min(max_tokens, 1024))
            retry_analysis = _json_loads_soft(raw_retry) or _analysis_from_jsonish(raw_retry)
            analysis = _merge_analysis(analysis, retry_analysis)

        queries, candidate_names = _queries_from_analysis(
            question,
            analysis,
            raw_analysis,
            max_candidates=max_candidates,
            max_search_queries=max_search_queries,
        )
        search_budget = max(5.0, time_budget_s - (time.monotonic() - total_start) - 15.0)
        t0 = time.monotonic()
        evidence = _search_many(queries, k, search_budget)
        elapsed = time.monotonic() - t0

        response = {
            "candidate_entities": candidate_names,
            "queries": queries,
            "evidence": evidence,
            "vision_analysis": analysis if analysis is not None else {"raw": _truncate(raw_analysis, 1500)},
            "search_elapsed_seconds": round(elapsed, 2),
            "usage_note": (
                "This is evidence only, not a final answer. Decide from the evidence yourself, "
                "or run focused web_search/wiki_search on the strongest remaining candidate or OCR clue."
            ),
        }
        return _compact_json(response, 7600)
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: visual_web_search failed: {_safe_error(exc)}"
