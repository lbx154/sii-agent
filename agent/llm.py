"""Unified LLM client. Switch between Azure OpenAI (AAD) and local vLLM via env."""
from __future__ import annotations
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_client():
    backend = os.getenv("LLM_BACKEND", "azure").lower()
    if backend == "azure":
        from openai import AzureOpenAI
        from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential()),
            "https://cognitiveservices.azure.com/.default",
        )
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_ad_token_provider=token_provider,
        )
        model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4")
        return client, model

    if backend == "vllm":
        from openai import OpenAI

        client = OpenAI(
            base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        )
        model = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")
        return client, model

    raise ValueError(f"Unknown LLM_BACKEND={backend}")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, minimum: int = 0, maximum: int = 1_000_000) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))


def _env_float(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _apply_env_sampling(payload: dict[str, Any]) -> None:
    if os.getenv("LLM_BACKEND", "azure").lower() != "vllm":
        return
    top_level = {
        "temperature": "SII_LLM_TEMPERATURE",
        "top_p": "SII_LLM_TOP_P",
        "presence_penalty": "SII_LLM_PRESENCE_PENALTY",
    }
    for key, env_name in top_level.items():
        value = _env_float(env_name)
        if value is not None and key not in payload:
            payload[key] = value
    extra_fields: dict[str, float | int] = {}
    top_k = os.getenv("SII_LLM_TOP_K")
    if top_k is not None and top_k.strip():
        try:
            extra_fields["top_k"] = int(top_k)
        except ValueError:
            pass
    for key, env_name in (
        ("min_p", "SII_LLM_MIN_P"),
        ("repetition_penalty", "SII_LLM_REPETITION_PENALTY"),
    ):
        value = _env_float(env_name)
        if value is not None:
            extra_fields[key] = value
    if extra_fields:
        extra_body = dict(payload.get("extra_body") or {})
        for key, value in extra_fields.items():
            extra_body.setdefault(key, value)
        payload["extra_body"] = extra_body


def _dynamic_context_limit() -> int:
    return _env_int("SII_LLM_CONTEXT_TOKEN_LIMIT", 0, minimum=0)


def _dynamic_clamp_enabled() -> bool:
    return _dynamic_context_limit() > 0 and _env_bool("SII_LLM_DYNAMIC_MAX_TOKENS", True)


def _tokenizer_path(model: str) -> str | None:
    explicit = os.getenv("SII_LLM_TOKENIZER_PATH")
    if explicit:
        return explicit
    candidates = [model]
    root = Path(__file__).resolve().parents[1]
    candidates.append(str(root / model))
    if model.startswith("Qwen3.5-9B"):
        candidates.append(str(root / "Qwen3.5-9B"))
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


@lru_cache(maxsize=4)
def _get_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            kind = str(item.get("type") or "").lower()
            if kind == "text":
                parts.append(str(item.get("text") or ""))
            elif kind in {"image_url", "image"}:
                parts.append("[image]")
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(part for part in parts if part)
    return str(content or "")


def _sanitized_messages(messages: list[dict]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": str(message.get("role") or "user"),
            "content": _content_text(message.get("content")),
        }
        for key in ("name", "tool_call_id", "tool_calls"):
            if key in message:
                item[key] = message[key]
        sanitized.append(item)
    return sanitized


def _estimate_prompt_tokens(payload: dict[str, Any], model: str) -> int | None:
    path = _tokenizer_path(model)
    if not path:
        return None
    try:
        tokenizer = _get_tokenizer(path)
        messages = _sanitized_messages(list(payload.get("messages") or []))
        tools = payload.get("tools")
        template_kwargs = {"tokenize": True, "add_generation_prompt": True}
        extra_body = payload.get("extra_body")
        chat_template_kwargs = {}
        if isinstance(extra_body, dict):
            chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        if "enable_thinking" in chat_template_kwargs:
            template_kwargs["enable_thinking"] = chat_template_kwargs["enable_thinking"]
        if tools:
            template_kwargs["tools"] = tools
        tokens = tokenizer.apply_chat_template(messages, **template_kwargs)
        if isinstance(tokens, dict):
            input_ids = tokens.get("input_ids") or []
            return int(len(input_ids))
        return int(len(tokens))
    except Exception:
        try:
            text = json.dumps(
                {
                    "messages": _sanitized_messages(list(payload.get("messages") or [])),
                    "tools": payload.get("tools") or [],
                    "tool_choice": payload.get("tool_choice"),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            tokenizer = _get_tokenizer(path)
            return int(len(tokenizer.encode(text, add_special_tokens=False))) + 32
        except Exception:
            return None


def _clamp_max_tokens(payload: dict[str, Any], model: str) -> None:
    if not _dynamic_clamp_enabled() or "max_tokens" not in payload:
        return
    try:
        requested = int(payload.get("max_tokens") or 0)
    except (TypeError, ValueError):
        return
    if requested <= 0:
        return
    prompt_tokens = _estimate_prompt_tokens(payload, model)
    if prompt_tokens is None:
        return
    margin = _env_int("SII_LLM_CONTEXT_TOKEN_MARGIN", 1, minimum=0, maximum=10000)
    available = _dynamic_context_limit() - prompt_tokens - margin
    clamped = max(1, available)
    if clamped < requested:
        payload["max_tokens"] = clamped


_CONTEXT_LIMIT_RE = re.compile(
    r"(?P<prompt>\d+)\s+tokens from the input messages\s+and\s+(?P<completion>\d+)\s+tokens for the completion",
    re.IGNORECASE,
)


def _retry_after_context_error(client: Any, payload: dict[str, Any], exc: Exception) -> Any:
    if not _dynamic_clamp_enabled() or "max_tokens" not in payload:
        raise exc
    match = _CONTEXT_LIMIT_RE.search(str(exc))
    if not match:
        raise exc
    prompt_tokens = int(match.group("prompt"))
    margin = _env_int("SII_LLM_CONTEXT_TOKEN_MARGIN", 1, minimum=0, maximum=10000)
    available = _dynamic_context_limit() - prompt_tokens - margin
    if available < 1:
        raise exc
    try:
        requested = int(payload.get("max_tokens") or 0)
    except (TypeError, ValueError):
        requested = 0
    if requested and available >= requested:
        raise exc
    retry_payload = dict(payload)
    retry_payload["max_tokens"] = available
    return client.chat.completions.create(**retry_payload)


def chat(messages: list[dict], tools: list[dict] | None = None, **kw) -> Any:
    client, model = get_client()
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = kw.pop("tool_choice", "auto")
        payload["parallel_tool_calls"] = kw.pop("parallel_tool_calls", False)
    payload.update(kw)
    _apply_env_sampling(payload)
    if os.getenv("LLM_BACKEND", "azure").lower() == "vllm" and os.getenv("VLLM_ENABLE_THINKING", "0").lower() in {"1", "true", "yes"}:
        extra_body = dict(payload.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", True)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        payload["extra_body"] = extra_body
    if os.getenv("LLM_BACKEND", "azure").lower() == "vllm" and os.getenv("VLLM_ENABLE_THINKING", "0").lower() not in {"1", "true", "yes"}:
        extra_body = dict(payload.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        payload["extra_body"] = extra_body
    _clamp_max_tokens(payload, model)
    try:
        return client.chat.completions.create(**payload)
    except Exception as exc:
        return _retry_after_context_error(client, payload, exc)
