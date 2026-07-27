"""Genie client – requires a Genie Space ID."""
from databricks_langchain.genie import GenieAgent

# !!! Replace with your actual Genie Space ID !!!
GENIE_SPACE_ID = "your-genie-space-id"

def ask_genie(question: str) -> str:
    """Ask a natural language question and get a structured answer."""
    agent = GenieAgent(space_id=GENIE_SPACE_ID)
    return agent.invoke(question)