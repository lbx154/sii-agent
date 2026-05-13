"""Final-answer tool — forces structured termination of the ReAct loop."""
from .registry import register


@register(
    "final_answer",
    "Submit the final answer when you are confident. "
    "After calling this, the agent loop terminates.",
    {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "concise final answer"},
            "rationale": {"type": "string", "description": "1–2 sentences justifying the answer"},
        },
        "required": ["answer"],
    },
)
def final_answer(answer: str, rationale: str = "") -> str:
    # The harness intercepts this; the return is informational only.
    return f"FINAL_ANSWER_SUBMITTED: {answer}"
