from __future__ import annotations

"""Vector Search retriever factory (Task 1.4 support / rag/store.py).

Returns a LangChain retriever over the Databricks Vector Search index. Reads
endpoint/index names from config.get_settings(). This exact retriever is reused
by the deployed model.
"""

from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from databricks.vector_search.client import VectorSearchClient

from config import get_settings

TEXT_COLUMN = "chunk_to_retrieve"
CITATION_COLUMNS = ["chunk_id", "source", "page"]


class DatabricksRetriever(BaseRetriever):
    """A LangChain retriever that queries a Databricks Vector Search index."""

    index: object
    columns: List[str]
    k: int = 4

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Perform similarity search and return LangChain Documents."""
        results = self.index.similarity_search(
            query_text=query,
            columns=self.columns,
            num_results=self.k,
        )
        data_array = results.get("result", {}).get("data_array", [])
        docs = []
        for row in data_array:
            # row is a list of values in the same order as self.columns
            text = row[0] if len(row) > 0 else ""
            metadata = {}
            for i, col in enumerate(self.columns[1:], start=1):
                metadata[col] = row[i] if i < len(row) else ""
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)


def get_retriever(k: int = 4) -> BaseRetriever:
    """Return a LangChain retriever with top‑k search."""
    settings = get_settings()
    vs_endpoint = settings["vs_endpoint"]
    vs_index = settings["vs_index"]

    client = VectorSearchClient()  # reads DATABRICKS_HOST/TOKEN from env
    index = client.get_index(endpoint_name=vs_endpoint, index_name=vs_index)

    return DatabricksRetriever(
        index=index,
        columns=[TEXT_COLUMN] + CITATION_COLUMNS,
        k=k,
    )