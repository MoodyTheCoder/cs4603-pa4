from __future__ import annotations


"""Full Document Analyst graph with Unity Catalog governed tools (Extra‑Credit Part 1)."""

from typing import Optional

from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

# Load environment variables from .env (needed for Databricks auth)
load_dotenv()

from databricks_langchain import UCFunctionToolkit

from agent.planner import make_planner
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer
from config import get_chat_llm
from rag.store import get_retriever


def load_uc_tools() -> list:
    """Return LangChain tools wrapping the registered UC functions."""
    toolkit = UCFunctionToolkit(
        function_names=[
            "main.default.compound_growth",
            "main.default.percent_change",
            "main.default.calculate",
        ]
    )
    return toolkit.tools


def build_graph(llm=None, retriever=None, tools=None):
    """Assemble the complete Document Analyst graph."""
    if llm is None:
        llm = get_chat_llm()
    if retriever is None:
        retriever = get_retriever()
    if tools is None:
        tools = load_uc_tools()                 # UC functions, not MCP

    planner_node = make_planner(llm)
    supervisor_node = make_supervisor(llm)
    rag_node = make_rag_agent(retriever, llm)
    mcp_node = make_mcp_node(tools, llm)        # same node, different tools
    synthesizer_node = make_synthesizer(llm)

    builder = StateGraph(AnalystState)
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node(RAG, rag_node)
    builder.add_node(MCP, mcp_node)
    builder.add_node(SYNTH, synthesizer_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
    )
    builder.add_edge(RAG, "supervisor")
    builder.add_edge(MCP, "supervisor")
    builder.add_edge(SYNTH, END)

    return builder.compile()



def make_mcp_node(tools, llm):
    """Node that executes a calculation step by calling a tool (UC function)."""
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
            # Find the tool
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


# Optional default instance – comment out if not needed
# default_graph = build_graph()