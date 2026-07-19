"""Planner node (Task 1.2).

TODO: Implement `make_planner(llm)` returning a node that:
  - reads the user question from state["messages"],
  - asks the LLM (PLANNER_PROMPT) for a JSON list of 2-5 steps,
  - parses it robustly (fallback to a single step on parse failure),
  - returns {"plan": [...], "current_step_index": 0, "step_results": []}.
"""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import PLANNER_PROMPT


def make_planner(llm):
    """Return a planner node that decomposes the user query into a list of steps."""
    def planner(state: dict) -> dict:
        messages = state.get("messages", [])

        user_query = ""
        for m in reversed(messages):
            if hasattr(m, "type") and m.type == "human":
                user_query = m.content
                break
        if not user_query:
            return {"plan": ["Please provide a question."], "current_step_index": 0, "step_results": []}

        response = llm.invoke([
            SystemMessage(content=PLANNER_PROMPT),
            HumanMessage(content=user_query)
        ])

        raw = response.content.strip()
        if raw.startswith("```json"):
            raw = raw[7:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        try:
            plan = json.loads(raw)
            if not isinstance(plan, list):
                raise ValueError("Not a list")
        except Exception:
            plan = [user_query]   

        return {
            "plan": plan,
            "current_step_index": 0,
            "step_results": [],
        }

    return planner