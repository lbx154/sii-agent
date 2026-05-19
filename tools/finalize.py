"""Final-answer tool — forces structured termination of the ReAct loop."""
from .registry import register


@register(
    "final_answer",
    "Submit the final answer only when the concise answer span is supported by current tool evidence. "
    "Include at least one supporting citation in the answer or rationale: "
    "a BrowseComp docid like [12345] or a source URL. After calling this, the agent loop terminates.",
    {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "concise final answer followed by at least one citation, e.g. 'Answer span [12345]' or 'Answer span [https://source]'",
            },
            "rationale": {"type": "string", "description": "1–2 sentences justifying the answer with docid or URL citations"},
        },
        "required": ["answer"],
    },
)
def final_answer(answer: str, rationale: str = "") -> str:
    # The harness intercepts this; the return is informational only.
    return f"FINAL_ANSWER_SUBMITTED: {answer}"
