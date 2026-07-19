"""Log, register, and serve the Document Analyst (Tasks 2.2 + 2.3).

Run:  uv run python deployment/deploy.py

TODO:
  - `log_and_register()`: set registry uri to 'databricks-uc', log the model via
    `mlflow.langchain.log_model(lc_model="deployment/agent_model.py", name=...,
    code_paths=[...], pip_requirements=[...], input_example={...})`, then
    `mlflow.register_model(...)` into $UC_CATALOG.$UC_SCHEMA.<model>.
  - `create_or_update_endpoint(uc_name, version)`: create/update a Model Serving
    endpoint with `WorkspaceClient().serving_endpoints`, workload_size='Small',
    scale_to_zero_enabled=True, and environment_vars supplied as secret refs
    ({{secrets/cs4603-deploy/...}}). Wait for READY and print the URL.
"""


from __future__ import annotations

import os
import time
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from dotenv import load_dotenv

load_dotenv()   # load .env variables if present

# ------------------------------------------------------------------
# Configuration (reads from environment)
# ------------------------------------------------------------------
CATALOG = os.environ.get("UC_CATALOG", "main")
SCHEMA = os.environ.get("UC_SCHEMA", "default")
MODEL_NAME = "document_analyst"                       # Unity Catalog model name
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
ENDPOINT_NAME = "pa4-doc-analyst-endpoint"            # serving endpoint name

# ------------------------------------------------------------------
# 1. Log and register the model in Unity Catalog
# ------------------------------------------------------------------
def log_and_register():
    """Log the model to MLflow (Databricks tracking) and register in Unity Catalog."""
    # Use Databricks tracking, not local
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Shared/pa4_document_analyst")

    with mlflow.start_run(run_name="pa4-deploy") as run:
        model_info = mlflow.langchain.log_model(
            lc_model="deployment/agent_model.py",
            artifact_path="agent",
            code_paths=[
                "agent",
                "rag",
                "tools",
                "config.py",
            ],
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
            input_example={
                "messages": [{"role": "user", "content": "What was the revenue in 2023?"}]
            },
        )

        registered = mlflow.register_model(
            model_uri=model_info.model_uri,
            name=FULL_MODEL_NAME,
        )
        print(f"✅ Model registered: {FULL_MODEL_NAME} version {registered.version}")
        return FULL_MODEL_NAME, registered.version


# ------------------------------------------------------------------
# 2. Create or update the serving endpoint
# ------------------------------------------------------------------
def create_or_update_endpoint(uc_name: str, version: str) -> str:
    """Create or update a Databricks Model Serving endpoint."""
    w = WorkspaceClient()

    # Environment variables for the serving container
    environment_vars = {
        "DATABRICKS_HOST": "{{secrets/cs4603-deploy/DATABRICKS_HOST}}",
        "DATABRICKS_TOKEN": "{{secrets/cs4603-deploy/DATABRICKS_TOKEN}}",
        "DATABRICKS_MODEL": "{{secrets/cs4603-deploy/DATABRICKS_MODEL}}",
        "VECTOR_SEARCH_ENDPOINT": os.environ["VECTOR_SEARCH_ENDPOINT"],
        "VECTOR_SEARCH_INDEX": os.environ["VECTOR_SEARCH_INDEX"],
        "EMBEDDINGS_ENDPOINT": os.environ.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en"),
    }

    served_entity = ServedEntityInput(
        entity_name=uc_name,
        entity_version=version,
        workload_size="Small",
        scale_to_zero_enabled=True,
        environment_vars=environment_vars,
    )

    # Check if endpoint already exists
    try:
        w.serving_endpoints.get(name=ENDPOINT_NAME)
        # Update existing endpoint
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
        )
        print(f"🔄 Updated endpoint '{ENDPOINT_NAME}' to version {version}")
    except Exception as e:
        if "does not exist" in str(e).lower() or "not found" in str(e).lower():
            # Create new endpoint
            config = EndpointCoreConfigInput(
                name=ENDPOINT_NAME,
                served_entities=[served_entity],
            )
            w.serving_endpoints.create(name=ENDPOINT_NAME, config=config)
            print(f"🚀 Created new endpoint '{ENDPOINT_NAME}'")
        else:
            raise

    # Wait for the endpoint to be READY
    print("⏳ Waiting for endpoint to become READY (this can take 10–20 minutes)...")
    while True:
        ep = w.serving_endpoints.get(name=ENDPOINT_NAME)
        state = ep.state.value if hasattr(ep.state, 'value') else ep.state
        if state == "READY":
            break
        if state in ("FAILED", "DEPLOYMENT_FAILED"):
            raise RuntimeError(f"Endpoint deployment failed: {ep.state_message}")
        time.sleep(30)

    host = os.environ["DATABRICKS_HOST"].rstrip("/")
    url = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    print(f"✅ Endpoint READY: {url}")
    return url


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    name, ver = log_and_register()
    url = create_or_update_endpoint(name, ver)
    print(f"🎉 Deployment complete. Endpoint URL: {url}")