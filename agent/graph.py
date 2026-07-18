from __future__ import annotations
"""Full Document Analyst graph (Tasks 1.5 + 1.7).

TODO:
  - `load_mcp_tools(server_path=None)`: connect the GIVEN MCP server over stdio
    (see langchain-mcp-adapters) and return its tools.
  - `make_mcp_node(tools, llm)`: execute one calculation step by letting the LLM
    call exactly one MCP tool, then append the result and increment the index.
  - `build_graph(llm=None, retriever=None, tools=None)`: assemble
    planner -> supervisor -> {rag_agent | mcp_tools} -> ... -> synthesizer.
    Inject dependencies so the graph can be unit-tested offline with fakes.
"""


import asyncio
import os
import sys
import threading
from typing import List, Optional

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import END, START, StateGraph

from agent.planner import make_planner
from agent.rag_agent import make_rag_agent
from agent.state import AnalystState
from agent.supervisor import MCP, RAG, SYNTH, make_supervisor, route_from_supervisor
from agent.synthesizer import make_synthesizer

from config import get_chat_llm
from rag.store import get_retriever

# ----------------------------------------------------------------------
# Background event loop for MCP tools (safe for serving containers)
# ----------------------------------------------------------------------
_mcp_loop: Optional[asyncio.AbstractEventLoop] = None


def _run_loop_forever(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _get_mcp_loop() -> asyncio.AbstractEventLoop:
    """Return a persistent background event loop, creating it once."""
    global _mcp_loop
    if _mcp_loop is None:
        if sys.platform == "win32":
            _mcp_loop = asyncio.ProactorEventLoop()
        else:
            _mcp_loop = asyncio.new_event_loop()
        threading.Thread(target=_run_loop_forever, args=(_mcp_loop,), daemon=True).start()
    return _mcp_loop


def _run_coro(coro, timeout: float | None = None):
    """Run an async coroutine on the background loop and return its result."""
    loop = _get_mcp_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def _wrap_sync(tool: BaseTool) -> BaseTool:
    """Make an async MCP tool callable synchronously (for LangGraph nodes)."""
    coroutine = tool.coroutine

    def sync_func(**kwargs):
        result = _run_coro(coroutine(**kwargs))
        # Handle different result shapes (MCP might return a tuple or an object with content)
        if isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], str):
            return result[0]
        if hasattr(result, "content") and isinstance(result.content, list):
            texts = [item.text for item in result.content if hasattr(item, "text")]
            if texts:
                return "\n".join(texts)
        return str(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=sync_func,
        coroutine=coroutine,
    )


def load_mcp_tools(server_path: str | None = None) -> List[BaseTool]:
    """Connect to the MCP server (stdio) and return sync‑wrapped tools."""
    global _mcp_client
    if server_path is None:
        server_path = os.path.join(os.path.dirname(__file__), "..", "tools", "mcp_server.py")
    if not os.path.exists(server_path):
        raise FileNotFoundError(f"MCP server not found: {server_path}")

    connections = {
        "analyst": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [server_path],
        }
    }

    async def _load_tools():
        client = MultiServerMCPClient(connections)
        tools = await client.get_tools()
        return client, tools

    client, raw_tools = _run_coro(_load_tools())
    _mcp_client = client            # keep alive so the subprocess stays open
    return [_wrap_sync(t) for t in raw_tools]

# ----------------------------------------------------------------------
# Graph assembly
# ----------------------------------------------------------------------
def build_graph(llm=None, retriever=None, tools=None):
    """Assemble the complete Document Analyst graph."""
    if llm is None:
        llm = get_chat_llm()
    if retriever is None:
        retriever = get_retriever()
    if tools is None:
        tools = load_mcp_tools()

    # Build node factories
    planner_node = make_planner(llm)
    supervisor_node = make_supervisor(llm)
    rag_node = make_rag_agent(retriever, llm)
    mcp_node = make_mcp_node(tools, llm)
    synthesizer_node = make_synthesizer(llm)

    # Wire the graph
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

def shutdown_mcp_tools() -> None:
    """Optional: stop the MCP client gracefully."""
    global _mcp_client
    if _mcp_client is not None:
        # The client might have a close() method; ignore for now
        _mcp_client = None

def make_mcp_node(tools, llm):
    """Node that executes a calculation step by calling an MCP tool."""
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


# ----------------------------------------------------------------------
# Default graph instance (used locally and by deployment/agent_model.py)
# ----------------------------------------------------------------------
default_graph = build_graph()