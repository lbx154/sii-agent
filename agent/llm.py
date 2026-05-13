"""Unified LLM client. Switch between Azure OpenAI (AAD) and local vLLM via env."""
from __future__ import annotations
import os
from functools import lru_cache
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


def chat(messages: list[dict], tools: list[dict] | None = None, **kw) -> Any:
    client, model = get_client()
    payload: dict[str, Any] = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = kw.pop("tool_choice", "auto")
        payload["parallel_tool_calls"] = kw.pop("parallel_tool_calls", False)
    payload.update(kw)
    if os.getenv("LLM_BACKEND", "azure").lower() == "vllm" and os.getenv("VLLM_ENABLE_THINKING", "0").lower() not in {"1", "true", "yes"}:
        extra_body = dict(payload.get("extra_body") or {})
        chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        payload["extra_body"] = extra_body
    return client.chat.completions.create(**payload)
