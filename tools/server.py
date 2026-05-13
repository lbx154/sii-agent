"""FastAPI tool sandbox exposing the registered tools over HTTP."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .registry import TOOL_REGISTRY, dispatch, tool_specs


app = FastAPI(title="sii-agent tool sandbox")


class ToolCall(BaseModel):
    name: str = Field(..., description="Registered tool name")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolBatch(BaseModel):
    calls: list[ToolCall] = Field(..., min_length=1, max_length=32)
    max_workers: int = Field(default=8, ge=1, le=16)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tools")
def tools() -> list[dict]:
    return tool_specs()


@app.post("/call")
def call_tool(call: ToolCall) -> dict[str, str]:
    if call.name not in TOOL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"unknown tool: {call.name}")
    return {"result": dispatch(call.name, call.arguments)}


@app.post("/call_many")
def call_many(batch: ToolBatch) -> dict[str, list[dict[str, Any]]]:
    """Run independent tool calls concurrently with bounded fan-out."""
    results: list[dict[str, Any] | None] = [None] * len(batch.calls)

    def run_one(index: int, call: ToolCall) -> dict[str, Any]:
        if call.name not in TOOL_REGISTRY:
            return {
                "index": index,
                "name": call.name,
                "ok": False,
                "error": f"unknown tool: {call.name}",
            }
        result = dispatch(call.name, call.arguments)
        response: dict[str, Any] = {
            "index": index,
            "name": call.name,
            "ok": not result.startswith("ERROR"),
            "result": result,
        }
        if result.startswith("ERROR"):
            response["error"] = result
        return response

    grouped_indices: dict[tuple[str, str], list[int]] = {}
    for index, call in enumerate(batch.calls):
        if call.name.startswith("browser_"):
            session_id = str(call.arguments.get("session_id") or "default")
            key = ("browser", session_id)
        else:
            key = ("stateless", str(index))
        grouped_indices.setdefault(key, []).append(index)

    def run_group(indices: list[int]) -> list[tuple[int, dict[str, Any]]]:
        return [(index, run_one(index, batch.calls[index])) for index in indices]

    with ThreadPoolExecutor(max_workers=min(batch.max_workers, len(batch.calls))) as pool:
        futures = {
            pool.submit(run_group, indices): key
            for key, indices in grouped_indices.items()
        }
        for future in as_completed(futures):
            for index, result in future.result():
                results[index] = result

    return {"results": [result for result in results if result is not None]}
