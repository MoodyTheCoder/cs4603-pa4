"""Corpus ingestion into Databricks Vector Search (Task 0.3 / rag/ingest.py).

Run inside a Databricks notebook (needs Spark + ai_parse_document/ai_prep_search).
Mirror PA2 Part 1:

TODO:
  - `build_chunks_table(spark, volume_path, chunks_table)`: parse the PDF with
    ai_parse_document, chunk with ai_prep_search into a Delta table with columns
    chunk_id, chunk_to_retrieve, chunk_to_embed, source, page. Enable Change Data
    Feed on the table.
  - `create_index()`: create a STANDARD Vector Search endpoint and a TRIGGERED
    Delta Sync index (primary_key='chunk_id',
    embedding_source_column='chunk_to_retrieve',
    embedding_model_endpoint_name=$EMBEDDINGS_ENDPOINT).
"""

from __future__ import annotations
import time
from typing import Optional
from databricks.ai_search.client import AISearchClient
from config import get_settings


def build_chunks_table(spark, volume_path: str, chunks_table: str) -> None:
    """
    Parse all PDFs in volume_path, chunk them with ai_prep_search,
    and insert into a Delta table with Change Data Feed enabled.
    """

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            chunk_id STRING,
            chunk_to_retrieve STRING,
            chunk_to_embed STRING,
            source_uri STRING,
            page INT
        ) TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)
    print(f"✅ Table {chunks_table} ready (Change Data Feed enabled).")

    spark.sql(f"""
        INSERT OVERWRITE {chunks_table}
        SELECT
            chunk.value:chunk_id::STRING AS chunk_id,
            chunk.value:chunk_to_retrieve::STRING AS chunk_to_retrieve,
            chunk.value:chunk_to_embed::STRING AS chunk_to_embed,
            prepped.path AS source_uri,
            CAST(chunk.value:pages[0].page_id::INT AS INT) AS page
        FROM (
            SELECT
                path,
                ai_prep_search(ai_parse_document(content)) AS result
            FROM read_files('{volume_path}/', format => 'binaryFile')
            WHERE path LIKE '%.pdf'
        ) prepped,
            LATERAL variant_explode(prepped.result:document.contents) AS chunk
    """)
    
    count = spark.table(chunks_table).count()
    print(f"✅ Inserted {count} chunks into {chunks_table}.")

def create_index(
    endpoint_name: Optional[str] = None,
    index_name: Optional[str] = None,
    chunks_table: Optional[str] = None,
    embedding_model: Optional[str] = None,
    wait_until_ready: bool = True,
    timeout_sec: int = 300
) -> dict:
    """
    Create a STANDARD Vector Search endpoint and a TRIGGERED Delta Sync index.
    Uses settings from environment variables (loaded via config.get_settings()).
    """
    settings = get_settings()
    
    if endpoint_name is None:
        endpoint_name = settings.get("VECTOR_SEARCH_ENDPOINT", "default-vs-endpoint")
    if index_name is None:
        catalog = settings.get("UC_CATALOG", "main")
        schema = settings.get("UC_SCHEMA", "default")
        index_name = f"{catalog}.{schema}.analyst_index"
    if chunks_table is None:
        catalog = settings.get("UC_CATALOG", "main")
        schema = settings.get("UC_SCHEMA", "default")
        chunks_table = f"{catalog}.{schema}.analyst_chunks"
    if embedding_model is None:
        embedding_model = settings.get("EMBEDDINGS_ENDPOINT", "databricks-gte-large-en")

    client = AISearchClient()

    existing_endpoints = [e["name"] for e in client.list_endpoints().get("endpoints", [])]
    if endpoint_name not in existing_endpoints:
        print(f"Creating endpoint '{endpoint_name}' (STANDARD)...")
        client.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print("✅ Endpoint created.")
    else:
        print(f"ℹ️ Endpoint '{endpoint_name}' already exists.")

    existing_indexes = [
        idx["name"] 
        for idx in client.list_indexes(name=endpoint_name).get("vector_indexes", [])
    ]
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}' (TRIGGERED)...")
        client.create_delta_sync_index(
            endpoint_name=endpoint_name,
            index_name=index_name,
            source_table_name=chunks_table,
            primary_key="chunk_id",
            pipeline_type="TRIGGERED",
            embedding_source_column="chunk_to_embed",
            embedding_model_endpoint_name=embedding_model,
            columns_to_sync=["chunk_to_retrieve", "source_uri", "page"]
        )
        print("✅ Index creation initiated.")
    else:
        print(f"ℹ️ Index '{index_name}' already exists.")

    status = "NOT_READY"
    if wait_until_ready:
        print(f"Waiting for index '{index_name}' to become READY... (timeout: {timeout_sec}s)")
        start = time.time()
        while time.time() - start < timeout_sec:
            info = client.get_index(endpoint_name=endpoint_name, index_name=index_name)
            status = info.get("status", {}).get("ready")
            if status:
                print("✅ Index is READY.")
                break
            print("⏳ Still provisioning... waiting 10s.")
            time.sleep(10)
        else:
            raise TimeoutError(f"Index '{index_name}' did not become READY within {timeout_sec} seconds.")

    return {
        "endpoint_name": endpoint_name,
        "index_name": index_name,
        "status": "READY" if status else "NOT_READY"
    }
    
def ingest_pipeline(spark, volume_path: str, chunks_table: str) -> None:
    """Run the full ingestion pipeline: build table + create index."""
    build_chunks_table(spark, volume_path, chunks_table)
    result = create_index(chunks_table=chunks_table)
    print(f"\n✅ Ingest complete. Endpoint: {result['endpoint_name']}, Index: {result['index_name']} ({result['status']})")