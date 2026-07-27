from __future__ import annotations

"""Full Document Analyst graph with UC tools + Genie structured retrieval (Extra‑Credit Part 2)."""

from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import END, START, StateGraph
from databricks_langchain import UCFunctionToolkit

from agent.planner import make_planner
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.synthesizer import make_synthesizer
from config import get_chat_llm
from rag.store import get_retriever

# ---------- Genie client ----------
from genie.genie_client import ask_genie

# ---------- Node name constants ----------
RAG = "rag_agent"
MCP = "mcp_tools"
SYNTH = "synthesizer"
GENIE = "genie_agent"

# ---------- UC tools (reused from Part 1) ----------
def load_uc_tools():
    toolkit = UCFunctionToolkit(
        function_names=[
            "main.default.compound_growth",
            "main.default.percent_change",
            "main.default.calculate",
        ]
    )
    return toolkit.tools

# ---------- Supervisor + routing (extended for Genie) ----------
def make_supervisor(llm):
    """Return a supervisor node that can route to rag_agent, mcp_tools, genie_agent, or synthesizer."""
    def supervisor(state: dict) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)
        if idx >= len(plan):
            return {"next_agent": SYNTH}

        step = plan[idx]
        # Heuristic keyword-based routing as fallback
        # (The LLM classification can be added for higher accuracy)
        lower = step.lower()
        if lower.startswith("retrieve:"):
            # If the step mentions tables, years, comparisons → route to genie
            if any(w in lower for w in ["year", "table", "trend", "compare", "highest", "lowest", "sql", "tabular"]):
                return {"next_agent": GENIE}
            return {"next_agent": RAG}
        elif lower.startswith("compute:"):
            return {"next_agent": MCP}
        # Fallback for unclear steps
        return {"next_agent": RAG}
    return supervisor

def route_from_supervisor(state: dict) -> str:
    return state.get("next_agent", SYNTH)

# ---------- Genie node ----------
def genie_node(state: dict) -> dict:
    """Answer a structured/tabular question using Genie."""
    plan = state.get("plan", [])
    idx = state.get("current_step_index", 0)
    if idx >= len(plan):
        return state
    step = plan[idx]
    answer = ask_genie(step)
    result_text = f"From Genie: {answer}"
    new_results = state.get("step_results", []) + [result_text]
    return {
        "step_results": new_results,
        "current_step_index": idx + 1,
    }

# ---------- Graph assembly ----------
def build_graph(llm=None, retriever=None, tools=None):
    if llm is None:
        llm = get_chat_llm()
    if retriever is None:
        retriever = get_retriever()
    if tools is None:
        tools = load_uc_tools()

    planner_node = make_planner(llm)
    supervisor_node = make_supervisor(llm)       # using the extended supervisor above
    rag_node = make_rag_agent(retriever, llm)
    mcp_node = make_mcp_node(tools, llm)
    synthesizer_node = make_synthesizer(llm)
    genie = genie_node

    builder = StateGraph(AnalystState)
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node(RAG, rag_node)
    builder.add_node(MCP, mcp_node)
    builder.add_node(GENIE, genie)
    builder.add_node(SYNTH, synthesizer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            RAG: RAG,
            MCP: MCP,
            GENIE: GENIE,
            SYNTH: SYNTH,
        },
    )
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(GENIE, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()

# ---------- Make MCP node (same as before) ----------
def make_mcp_node(tools, llm):
    llm_with_tools = llm.bind_tools(tools)
    MCP_SYSTEM_PROMPT = """\
You are a calculation assistant with access to precise math tools.
For the given step, choose exactly one tool and call it with the correct arguments.\
"""
    def mcp_tools(state: dict) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)
        if idx >= len(plan):
            return state
        step = plan[idx]
        response = llm_with_tools.invoke([
            {"role": "system", "content": MCP_SYSTEM_PROMPT},
            {"role": "user", "content": step},
        ])
        if not response.tool_calls:
            result = f"Unable to process step: {step}"
        else:
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_obj = next((t for t in tools if t.name == tool_name), None)
            if tool_obj is None:
                result = f"Tool '{tool_name}' not found."
            else:
                try:
                    result = str(tool_obj.invoke(tool_args))
                except Exception as e:
                    result = f"Tool error: {e}"
        new_results = state.get("step_results", []) + [result]
        return {
            "step_results": new_results,
            "current_step_index": idx + 1,
        }
    return mcp_tools

# Optional default graph
# default_graph = build_graph()