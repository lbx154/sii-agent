"""Quick smoke test — does the LLM client work + does ReAct produce a final answer?"""
from agent.runner import run_baseline
from harness.controller import HarnessConfig

if __name__ == "__main__":
    out = run_baseline(
        "Who wrote the novel 'Norwegian Wood' and in what year was it first published?",
        cfg=HarnessConfig(max_steps=6, max_wall_seconds=90),
    )
    r = out.result
    print("STOP:", r.stop_reason)
    print("STEPS:", r.steps, "TOOL_CALLS:", r.tool_calls, f"ELAPSED:{r.elapsed:.1f}s")
    print("ANSWER:", r.final_answer)
    print("RATIONALE:", r.rationale)
