"""RAG agent node (Task 1.4) — retrieves from Databricks Vector Search.

TODO: Implement `make_rag_agent(retriever, llm)` returning a node that:
  - retrieves top-k chunks for the current step,
  - formats them with [source: file, p.N] citations,
  - extracts a single cited fact via the LLM (or 'not found in documents'),
  - appends the fact to step_results and increments current_step_index.
Reuse `rag/store.py::get_retriever()` so local and deployed retrieval match.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from agent.prompts import RAG_EXTRACT_PROMPT


def format_docs(docs) -> str:
    """Format retrieved documents into a single string with citations."""
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown file")
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content.strip()
        parts.append(f"[{i}] (source: {src}, p.{page})\n{content}")
    return "\n\n".join(parts)

def make_rag_agent(retriever, llm):
    """Return a RAG agent node that retrieves and extracts a fact for one step."""

    def rag_agent(state: dict) -> dict:
        plan = state.get("plan", [])
        idx = state.get("current_step_index", 0)

        if idx >= len(plan):
            return state

        step = plan[idx]

        docs = retriever.invoke(step)
        if not docs:
            result_text = "Not found in documents."
        else:
            formatted = format_docs(docs)
            prompt = RAG_EXTRACT_PROMPT.format(question=step, documents=formatted)
            response = llm.invoke([HumanMessage(content=prompt)])
            extracted = response.content.strip()
            if "not found" in extracted.lower():
                result_text = "Not found in documents."
            else:
                result_text = extracted

        new_results = state.get("step_results", []) + [result_text]
        return {
            "step_results": new_results,
            "current_step_index": idx + 1,
        }

    return rag_agent