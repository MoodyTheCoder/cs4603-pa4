"""Bonus B — deploy via the databricks-agents SDK (deployment/deploy_agents.py).

TODO: Log + register the model (reuse the pattern from deploy.py), then call
`databricks.agents.deploy(model_name=..., model_version=...)` to provision the
serving endpoint AND the Review App in one call. Print the endpoint + review URL.
"""

from __future__ import annotations

import os
import mlflow
from databricks import agents
from dotenv import load_dotenv
from mlflow.pyfunc import PythonModel, PythonModelContext
import pandas as pd

load_dotenv()

CATALOG = os.environ["UC_CATALOG"]
SCHEMA = os.environ["UC_SCHEMA"]
MODEL_NAME = "document_analyst_agent_bonusb"   # a separate name to avoid conflict
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"


class GraphWrapper(PythonModel):
    """Wraps the LangGraph agent to return only the final answer string."""

    def predict(self, context: PythonModelContext, model_input: pd.DataFrame) -> str:
        # model_input is expected to have a column 'messages' with a list of dicts
        from agent.graph import build_graph
        from langchain_core.messages import HumanMessage

        # Rebuild graph inside the container
        graph = build_graph()

        # Get the first row's messages
        messages = model_input.iloc[0]["messages"]
        # Convert to HumanMessage if they are plain dicts
        if isinstance(messages, list) and len(messages) > 0:
            first_msg = messages[0]
            if isinstance(first_msg, dict):
                human_msg = HumanMessage(content=first_msg["content"])
            else:
                human_msg = first_msg
        else:
            human_msg = HumanMessage(content="")

        state = {"messages": [human_msg]}
        result = graph.invoke(state)
        return result.get("final_answer", "")


def main() -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Shared/pa4_document_analyst")

    with mlflow.start_run(run_name="bonus-b-deploy"):
        # Log the wrapper model (not the raw LangGraph) to get a simple string output
        model_info = mlflow.pyfunc.log_model(
            artifact_path="agent",
            python_model=GraphWrapper(),
            code_paths=["agent", "rag", "tools", "config.py"],
            pip_requirements=[
                "mlflow>=2.16.0",
                "langgraph>=0.2.0",
                "langchain>=0.3.0",
                "langchain-core>=0.3.0",
                "langchain-openai>=0.2.0",
                "databricks-langchain>=0.1.0",
                "databricks-vectorsearch>=0.40",
                "databricks-sdk>=0.23.0",
                "mcp>=1.4.1",
                "langchain-mcp-adapters<0.0.5",
                "openai>=1.40.0",
                "python-dotenv>=1.0.0",
                "httpx>=0.27.0",
                "pydantic>=2.0.0",
            ],
            input_example=pd.DataFrame([{"messages": [{"role": "user", "content": "What was the revenue?"}]}]),
        )

        registered = mlflow.register_model(model_info.model_uri, FULL_MODEL_NAME)
        print(f"✅ Registered model version: {registered.version}")

        deployment = agents.deploy(
            model_name=FULL_MODEL_NAME,
            model_version=registered.version,
            scale_to_zero=True,
        )
        print(f"🎉 Endpoint: {deployment.endpoint_name}")
        print(f"📝 Review App: {deployment.review_app_url}")


if __name__ == "__main__":
    main()