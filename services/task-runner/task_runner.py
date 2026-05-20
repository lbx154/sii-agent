"""
Qwen Agent Harness — Main Orchestrator
======================================

Drives the agent loop:
  1. Writes system + user turn to trajectory
  2. Calls Qwen-3.5 via Sglang OpenAI-compat API
  3. Dispatches tool calls and records results
  4. Loops until finish_reason == 'stop' or max_steps reached

Usage (CLI):
    python -m task_runner \
        --instruction "请帮我查询上海创智学院谢源老师的相关信息，并获取其代表作。" \
        --task-id my_task_010
    python -m task_runner \
        --instruction "请先帮我分析图像的内容，再调用search_image工具进行图像搜索。" \
        --image "/inspire/qb-ilm2/project/26summer-camp-01/qiaojingyang-240208120192/harness-sii/datasets/simpleVQA/CCSimpleQA/0.jpg" \
        --image-url "https://datasets-server.huggingface.co/cached-assets/ohjoonhee/SimpleVQA/--/8fefe22e2775a6ac0a73ac22edba8a01536b8a59/--/default/test/0/image/image.jpg?Expires=1779081093&Signature=cHN23HVLSGpna8jlbFRnpt90RruGsgAjpRTot1IArVYgZrUFTz2Fl5Gn7OSU6QVmxQMZFc8csXss9g9-8sh9fAPpRbOAwgdlVdH8yg1fr4pIGLneUXz8swhhSlSECAbYyDi-r2we7kizYjnuvlfDa45BsRU32c7sPVLttqVWbNH8vWrYi9rTajYAdbCn9l2zYMN~zpSp~8b4T2OwMGw6feZl3fBdZxMPWmuyf2GTaIAiisDTQd2b6-8Yq3CsIzjfmW6M4nN0T5O8FXLR-yTd5ve9Pj40U13410vyqUbcOGDC~R7hCtrXDhxpg4aivRPLcjcHPTbKgu10K09cWSTZAQ__&Key-Pair-Id=K204OQ5RWQVDLD" \
        --task-id my_task_011
"""

import argparse
import json
import logging
import os
import uuid
from typing import Optional
from pathlib import Path

from openai import OpenAI

from roles import Role
from trajectory import Trajectory
from tools.search_tool import search_text, search_image
from tools.browser_tool import (
    browser_navigate, browser_get_text, browser_click,
    browser_type, browser_parallel,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("harness.task_runner")

# ---------------------------------------------------------------------------
# LLM connection (Sglang OpenAI-compat, two nodes behind Nginx)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "qwen-3.5")
MAX_STEPS    = int(os.getenv("MAX_STEPS", "20"))
MAX_TOKENS   = int(os.getenv("MAX_TOKENS", "16000"))

# 调试开关：True = 不向 LLM 注册 tools，纯文本对话，便于先验证 LLM 通路
# 工具实现接好后默认关闭；如需调试 LLM 通路，export DISABLE_TOOLS=1
DISABLE_TOOLS = os.getenv("DISABLE_TOOLS", "0") == "1"

# ---------------------------------------------------------------------------
# Tool schema (OpenAI function-calling format)
# ---------------------------------------------------------------------------
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": (
                "基于 Serper (Google) 的联网文字搜索，并用 Jina Reader 抽取每个结果页面的正文"
                "返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":     {"type": "string",  "description": "搜索关键词"},
                    "top_k":     {"type": "integer", "description": "返回条数（1-3）", "default": 1},
                    "fetch":     {"type": "boolean", "description": "是否抓取正文，false 时只返回摘要", "default": True},
                    "max_chars": {"type": "integer", "description": "每篇正文截断的最大字符数", "default": 500},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_image",
            "description": (
                "图搜文：基于 Google Lens (Serper /lens) 的反向图像搜索，并用 "
                "Jina Reader 抽取结果页面正文。输入必须是 http(s) 图片 URL 。"
                "返回 [{rank,title,url,snippet,content}]。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string",  "description": "图片的 http(s) URL"},
                    "top_k":     {"type": "integer", "description": "返回条数（1-3）", "default": 1},
                    "fetch":     {"type": "boolean", "description": "是否抓取正文", "default": True},
                    "max_chars": {"type": "integer", "description": "每篇正文截断的最大字符数", "default": 500},
                },
                "required": ["image_url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": (
                "在沙盒浏览器中打开一个 URL。默认顺带返回前若干字符的页面文本预览，"
                "需要完整正文请再调 browser_get_text。返回 "
                "{ok,url,title,wait_until,text_preview?,truncated?}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url":          {"type": "string",  "description": "要访问的 URL（可省略协议头）"},
                    "wait_until":   {"type": "string",  "description": "Playwright 等待策略",
                                     "enum": ["domcontentloaded", "load", "networkidle"],
                                     "default": "domcontentloaded"},
                    "include_text": {"type": "boolean", "description": "是否返回 text_preview", "default": True},
                    "max_text":     {"type": "integer", "description": "text_preview 字符上限", "default": 2000},
                    "timeout":      {"type": "integer", "description": "导航超时秒数", "default": 30},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "返回当前页面清洗后的可见文本。返回 {ok,url,title,text,truncated,total_chars}。",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {"type": "integer", "description": "正文最大字符数", "default": 5000},
                    "timeout":   {"type": "integer", "description": "抽取超时秒数", "default": 15},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": (
                "用 CSS 选择器点击当前页的元素。selector 接受任意合法 CSS，例如 "
                "'#login', 'button.primary', \"button:has-text('确定')\"。返回 "
                "{ok,selector,current_url,current_title,navigated}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string",  "description": "CSS 选择器"},
                    "nth":      {"type": "integer", "description": "命中多个时取第几个（0 表示用 .first）", "default": 0},
                    "timeout":  {"type": "integer", "description": "点击超时秒数", "default": 10},
                },
                "required": ["selector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": (
                "向一个 CSS 选择器选中的输入框键入文本，可选按回车提交。"
                "返回 {ok,selector,submitted,current_url,current_title}。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string",  "description": "CSS 选择器（输入框）"},
                    "text":     {"type": "string",  "description": "要输入的文本"},
                    "submit":   {"type": "boolean", "description": "输入完是否按 Enter", "default": False},
                    "clear":    {"type": "boolean", "description": "输入前是否清空字段", "default": True},
                    "timeout":  {"type": "integer", "description": "操作超时秒数", "default": 10},
                },
                "required": ["selector", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_parallel",
            "description": (
                "在沙盒浏览器中**并发**打开多个 URL。"
                "mode='navigate' 每个返回 {url,title,text_preview,truncated}；"
                "mode='get_text' 每个返回 {url,title,text,truncated,total_chars}。"
                "返回值是一个列表，单个 URL 失败不影响其他。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "urls":            {"type": "array", "items": {"type": "string"}, "description": "URL 列表"},
                    "mode":            {"type": "string", "enum": ["navigate", "get_text"], "default": "navigate"},
                    "max_chars":       {"type": "integer", "description": "每条结果文本上限；缺省时 navigate=2000，get_text=5000"},
                    "wait_until":      {"type": "string",
                                        "enum": ["domcontentloaded", "load", "networkidle"],
                                        "default": "domcontentloaded"},
                    "max_concurrency": {"type": "integer", "description": "同时打开的标签页数（1-8）", "default": 4},
                    "timeout":         {"type": "integer", "description": "单页超时秒数", "default": 30},
                },
                "required": ["urls"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool function dispatch map
# ---------------------------------------------------------------------------
TOOL_FN_MAP = {
    "search_text":      lambda a: search_text(**a),
    "search_image":     lambda a: search_image(**a),
    "browser_navigate": lambda a: browser_navigate(**a),
    "browser_get_text": lambda a: browser_get_text(**a),
    "browser_click":    lambda a: browser_click(**a),
    "browser_type":     lambda a: browser_type(**a),
    "browser_parallel": lambda a: browser_parallel(**a),
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个高效、严谨的任务执行 Agent，运行在配备多工具的自动化框架中。

## 行为准则
1. 每一步先在 <think>...</think> 标签中简述推理，再决定调用工具或直接回答。
2. 任务完成后输出清晰的最终答案，无需再调用工具。
3. 若工具返回 ok=False，分析 error，最多重试 2 次同类操作；仍失败则换工具或方法。
4. 若调用search_image工具，请使用输入图像的在线链接。
5. 每一步要不调用工具，要不输出最终答案，不能同时输出空的工具调用或者空的内容。
"""


# ---------------------------------------------------------------------------
# Core run_task function
# ---------------------------------------------------------------------------

def run_task(
    task: dict,
    max_steps: int = MAX_STEPS,
    llm_base_url: str = LLM_BASE_URL,
    model_name: str = MODEL_NAME,
    trajectory_dir: str = "trajectories",
) -> dict:
    """
    Execute a task with the Qwen agent loop.

    Args:
        task:            Dict with keys:
                           - "instruction" (str, required): task description
                           - "id"          (str, optional): task identifier
                           - "image_b64"   (str, optional): base64 image for vision input
                           - "image_url"   (str, optional): online image url for vision input
        max_steps:       Maximum agent loop iterations.
        llm_base_url:    Sglang / OpenAI-compat endpoint.
        model_name:      Model identifier served by Sglang.
        trajectory_dir:  Directory to write JSONL trajectories.

    Returns:
        Dict with keys: task_id, answer, steps, trajectory_path, summary
    """
    task_id     = task.get("id") or str(uuid.uuid4())[:8]
    instruction = task["instruction"]
    image_b64   = task.get("image_b64")
    image_url   = task.get("image_url")

    logger.info("run_task: task_id=%s", task_id)

    traj   = Trajectory(task_id, output_dir=trajectory_dir)
    client = OpenAI(base_url=llm_base_url, api_key="EMPTY")

    # ------------------------------------------------------------------ step 0
    # Write system turn
    traj.write(Role.SYSTEM, SYSTEM_PROMPT, step_id=0)

    # Build user message (optionally include image)
    if image_b64 and image_url:
        user_content = [
            {"type": "text",      "text": instruction + "输入图像的在线链接：" + image_url},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]
    else:
        user_content = instruction

    traj.write(Role.USER, user_content, step_id=0)

    # ------------------------------------------------------------------ loop
    final_answer = ""

    for step in range(1, max_steps + 1):
        logger.info("--- step %d ---", step)

        messages = traj.to_messages()
        logger.info("messages count=%d, sending to LLM ...", len(messages))

        # 构造请求参数：调试模式下不注册 tools，避免协议不匹配
        request_kwargs = dict(
            model=model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            extra_body={"enable_thinking": True},
        )
        if not DISABLE_TOOLS:
            request_kwargs["tools"] = TOOLS_SCHEMA
            request_kwargs["tool_choice"] = "auto"

        try:
            response = client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            logger.error("LLM call failed: %s", exc, exc_info=True)
            traj.write(
                Role.TOOL,
                f"[HARNESS ERROR] LLM call failed at step {step}: {exc}",
                step_id=step,
            )
            break

        choice  = response.choices[0]
        msg     = choice.message
        print(msg)
        content = msg.content or ""
        reasoning_content = msg.reasoning_content or ""
        total_tokens = response.usage.total_tokens or ""

        # 调试模式下强制忽略 tool_calls（虽然不传 tools 通常不会出现）
        tool_calls = None if DISABLE_TOOLS else msg.tool_calls

        # Write assistant turn
        tool_calls_data = (
            [tc.model_dump() for tc in tool_calls]
            if tool_calls else []
        )
        
        extra = {}
        
        if tool_calls_data:
            extra["tool_calls"] = tool_calls_data
        if reasoning_content:
            extra["reasoning_content"] = reasoning_content
        if total_tokens:
            extra["total_tokens"] = total_tokens
                        
        traj.write(
            Role.ASSISTANT,
            content,
            step_id=step,
            extra= extra if extra else None,
        )

        if content:
            logger.info("assistant: %s", content[:200])
        logger.info("finish_reason=%s, has_tool_calls=%s", choice.finish_reason, bool(tool_calls))

        # Done?
        # 标准退出条件：没有 tool_calls 时就结束（finish_reason 可能是 stop / length 等）
        if not tool_calls and choice.finish_reason and content != "":
            final_answer = content
            logger.info("Task complete at step %d", step)
            break
        
        if not tool_calls and content == "":
            continue

        # -------------------------------------------------------- tool calls
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                fn_args = {}
                logger.warning("Bad tool args JSON: %s", exc)

            logger.info("tool_call: %s(%s)", fn_name, fn_args)

            # Dispatch
            if fn_name not in TOOL_FN_MAP:
                tool_result = f"[ERROR] Unknown tool: {fn_name}"
            else:
                try:
                    raw = TOOL_FN_MAP[fn_name](fn_args)
                    # 工具返回结构化对象时，序列化为 JSON 字符串方便 LLM 解读
                    if isinstance(raw, (dict, list)):
                        tool_result = json.dumps(raw, ensure_ascii=False)
                    else:
                        tool_result = str(raw)
                except Exception as exc:
                    tool_result = f"[ERROR] Tool '{fn_name}' raised: {type(exc).__name__}: {exc}"
                    logger.exception("Tool error")

            logger.info("tool_result (%s): %s", fn_name, str(tool_result)[:200])

            traj.write(
                Role.TOOL,
                tool_result,
                step_id=step,
                tool_call_id=tc.id,
                extra={"fn_name": fn_name, "fn_args": fn_args},
            )
    else:
        logger.warning("Reached max_steps=%d without finish_reason=stop", max_steps)
        final_answer = "[HARNESS] Max steps reached. Last assistant message above."

    summary = traj.summary()
    logger.info("Trajectory summary: %s", summary)

    return {
        "task_id":         task_id,
        "answer":          final_answer,
        "steps":           step,
        "trajectory_path": str(traj.path),
        "summary":         summary,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen Agent Harness — run a single task from the command line",
    )
    p.add_argument("--instruction", "-i", required=True, help="Task instruction text")
    p.add_argument("--task-id",     "-t", default=None,  help="Optional task ID (auto-generated if omitted)")
    p.add_argument("--max-steps",   "-s", type=int, default=MAX_STEPS, help="Max agent loop steps")
    p.add_argument("--llm-url",           default=LLM_BASE_URL, help="Sglang base URL")
    p.add_argument("--model",             default=MODEL_NAME,   help="Model name")
    p.add_argument("--traj-dir",          default="trajectories", help="Trajectory output directory")
    p.add_argument("--image",             default=None, help="Local path to input image (optional)")
    p.add_argument("--image-url",         default=None, help="Online path to input image (optional)")
    return p.parse_args()


if __name__ == "__main__":
    import base64

    args = _parse_args()

    image_b64 = None
    if args.image:
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
    image_url = None
    if args.image_url:
        image_url = args.image_url

    task = {
        "instruction": args.instruction,
        "image_b64":   image_b64,
        "image_url":   image_url,
    }
    if args.task_id:
        task["id"] = args.task_id

    result = run_task(
        task,
        max_steps=args.max_steps,
        llm_base_url=args.llm_url,
        model_name=args.model,
        trajectory_dir=args.traj_dir,
    )

    print("\n" + "=" * 60)
    print("TASK COMPLETE")
    print("=" * 60)
    print(f"Task ID:  {result['task_id']}")
    print(f"Steps:    {result['steps']}")
    print(f"Traj:     {result['trajectory_path']}")
    print(f"\nAnswer:\n{result['answer']}")
