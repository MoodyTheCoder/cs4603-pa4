from __future__ import annotations

"""Supervisor node + routing edge (Task 1.3).

TODO:
  - `make_supervisor(llm)`: if current_step_index >= len(plan) -> next_agent =
    'synthesizer'; else classify the current step to 'rag_agent' or 'mcp_tools'.
  - `route_from_supervisor(state)`: return state["next_agent"] for the
    conditional edge.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import SUPERVISOR_PROMPT

RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"


def make_supervisor(llm):
    """Return a supervisor node that routes to the correct agent."""

    def supervisor(state: dict) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        if idx >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[idx]

        response = llm.invoke([
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=f"Current step: {step}")
        ])
        decision = response.content.strip().lower()

        if decision in {RAG, MCP, SYNTH}:
            return {"next_agent": decision}

        if step.lower().startswith("retrieve"):
            return {"next_agent": RAG}
        elif step.lower().startswith("compute"):
            return {"next_agent": MCP}
        else:
            return {"next_agent": RAG}   

    return supervisor


def route_from_supervisor(state: dict) -> str:
    """Conditional edge function."""
    return state.get("next_agent", SYNTH)