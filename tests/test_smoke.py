"""Offline smoke test for the Document Analyst graph (Bonus A test target).

This is the target the Bonus A CI pipeline runs to prove the graph wires up
before any deploy. Fill it in once your nodes are implemented.

TODO (Task 1.7 / Bonus A):
  - Build fake LLM / retriever / tool objects (no Databricks, no network).
  - Call `build_graph(llm=FakeLLM(), retriever=FakeRetriever(), tools=[FakeTool()])`.
  - Invoke it on a combined retrieval+calculation query and assert that a plan was
    produced, both specialists ran, and the final answer surfaced on messages[-1].

Run:  uv run pytest -q
"""
from __future__ import annotations

import os
import sys

# Ensure the project root is in the path (harmless safety net)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage


def test_graph_module_imports():
    """Minimal collection guard: the graph module must import cleanly."""
    from agent.graph import build_graph  # noqa: F401


def test_graph_compiles_and_returns_messages():
    """Build the graph with mocked LLM, retriever, and MCP tools, then invoke."""
    # 1. Fake LLM – the order of calls is:
    #    planner, supervisor (×3), synthesizer
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = [
        AIMessage(content='["Retrieve: revenue", "Compute: 10% increase"]'),  # planner
        AIMessage(content="rag_agent"),   # supervisor step 0
        AIMessage(content="mcp_tools"),   # supervisor step 1
        AIMessage(content="synthesizer"), # supervisor step 2
        AIMessage(content="Final answer."), # synthesizer
    ]

    # 2. Fake retriever – returns a dummy document with metadata
    fake_retriever = MagicMock()
    fake_doc = MagicMock()
    fake_doc.page_content = "Revenue: 16.91 trillion"
    fake_doc.metadata = {"source": "test.pdf", "page": "1"}
    fake_retriever.invoke.return_value = [fake_doc]

    # 3. Mock MCP tool loading to avoid starting a real subprocess
    with patch("agent.graph.load_mcp_tools", return_value=[]):
        from agent.graph import build_graph
        graph = build_graph(llm=fake_llm, retriever=fake_retriever, tools=[])

    # 4. Invoke the graph
    result = graph.invoke({"messages": [HumanMessage(content="test query")]})

    # 5. Assertions
    assert "messages" in result
    assert len(result["messages"]) > 0, "Messages should contain at least the final answer"
    last_msg = result["messages"][-1]
    assert hasattr(last_msg, "content")
    assert last_msg.content == "Final answer."
    assert result.get("final_answer") != "", "final_answer must not be empty"