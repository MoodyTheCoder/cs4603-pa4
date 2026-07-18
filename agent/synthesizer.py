
"""Synthesizer node (Task 1.6).

TODO: Implement `make_synthesizer(llm)` returning a node that combines
step_results into one cited answer and writes it to BOTH `final_answer` AND
the `messages` channel as an AIMessage (required for the OpenAI-compatible
serving contract — see spec Task 1.6).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from agent.prompts import SYNTHESIZER_PROMPT


def make_synthesizer(llm):
    """Return a synthesizer node that combines step results into a final answer."""

    def synthesizer(state: dict) -> dict:
        user_query = ""
        for msg in state.get("messages", []):
            if hasattr(msg, "type") and msg.type == "human":
                user_query = msg.content
                break

        step_results = state.get("step_results", [])
        context = "\n".join(
            f"Step {i+1}: {r}" for i, r in enumerate(step_results)
        )

        response = llm.invoke([
            {"role": "system", "content": SYNTHESIZER_PROMPT},
            {"role": "user", "content": f"Original question: {user_query}\n\nStep results:\n{context}"}
        ])
        answer = response.content.strip()

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],   # critical for deployment
        }

    return synthesizer