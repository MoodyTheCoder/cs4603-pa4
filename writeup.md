### Part 1 – Governed Tools

**Why does declaring `DatabricksFunction(...)` as a resource remove the need for a token in your deployment code?**  
When you include a `DatabricksFunction` resource in the `mlflow.langchain.log_model(..., resources=[...])` call, the Databricks platform automatically grants the serving endpoint’s internal identity the `EXECUTE` privilege on the specified functions. At deployment time, short‑lived credentials are provisioned and injected into the endpoint container, so the model code can call the UC functions without a manually‑passed secret token. This is a declarative, governed authorization model – you describe what resources the endpoint needs, and the platform handles access.


### Part 2 – Structured Retrieval (Genie)

1. **When would you use Genie (text‑to‑SQL) instead of Vector Search?**  
   Genie is ideal for structured, numerical queries that require exact calculations or comparisons (e.g., “which year had the highest revenue?”). It executes SQL on governed tables, returning precise numbers. Vector Search is better for narrative, descriptive information found in unstructured text.

2. **How does the supervisor decide which node to route to?**  
   The supervisor first checks if the step starts with "Retrieve:". If it contains tabular keywords (year, table, trend, etc.) it routes to `genie_agent`; otherwise to `rag_agent`. Steps starting with "Compute:" are sent to `mcp_tools`. This is a keyword‑based heuristic that can be enhanced with LLM classification.

3. **What are the tradeoffs of using Genie vs. RAG for structured data?**  
   Genie provides precise, reproducible answers but requires a well‑curated table and a Genie Space (or SQL warehouse). RAG is more flexible for mixed content but may miss exact numerical relationships unless the data is textually present. Genie adds governance and lineage benefits.