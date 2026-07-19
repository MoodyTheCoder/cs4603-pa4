"""MLflow models-from-code definition (Task 2.1).

TODO: Make this file self-contained so MLflow can serialise it:
  - validate DATABRICKS_HOST/TOKEN/MODEL at import time (clear error if missing),
  - rebuild the graph with production clients (LLM, Vector Search retriever,
    MCP tools),
  - end with `mlflow.models.set_model(graph)`.

Must import cleanly:  python -c "import deployment.agent_model"
"""
from __future__ import annotations

import os
import sys
import mlflow
from dotenv import load_dotenv

load_dotenv()

REQUIRED_ENV_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_MODEL",
    "EMBEDDINGS_ENDPOINT",
    "VECTOR_SEARCH_ENDPOINT",
    "VECTOR_SEARCH_INDEX",
]

missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
if missing:
    raise EnvironmentError(
        f"Missing required environment variables: {', '.join(missing)}\n"
        "Set them in the serving endpoint environment_vars as secret references."
    )

try:
    from agent.graph import build_graph, load_mcp_tools
    from config import get_chat_llm
    from rag.store import get_retriever
except ImportError as e:
    raise ImportError(
        "Failed to import agent modules. Ensure code_paths includes 'agent', 'rag', 'tools', 'config.py'."
    ) from e

llm = get_chat_llm(temperature=0.0)
retriever = get_retriever(k=4)

tools = load_mcp_tools()

graph = build_graph(
    llm=llm,
    retriever=retriever,
    tools=tools,
)

mlflow.models.set_model(graph)

print("✅ agent_model: graph loaded and set for MLflow.")