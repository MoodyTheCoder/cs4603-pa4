# Extra Credit Assignment 1 — Writeup

## Part 1 – Governed Tools with Unity Catalog Functions

**Why does declaring `DatabricksFunction(...)` as a resource remove the need for a token in your deployment code?**  
When you include a `DatabricksFunction` resource in `mlflow.langchain.log_model(..., resources=[...])`, the Databricks platform automatically grants the serving endpoint’s internal identity the `EXECUTE` privilege on the specified functions. At deployment time, short‑lived credentials are provisioned and injected into the endpoint container, so the model code can call the UC functions without a manually‑passed secret token. This is a declarative, governed authorization model – you describe what resources the endpoint needs, and the platform handles access.

---

## Part 2 – Structured Retrieval with AI/BI Genie

**1. When would you use Genie (text‑to‑SQL) instead of Vector Search?**  
Genie is ideal for structured, numerical queries that require exact calculations or comparisons (e.g., “which year had the highest net income?” or “show me the year‑over‑year revenue growth”). It executes SQL on governed Delta tables, returning precise numbers. Vector Search is better for narrative, descriptive information found in unstructured text (like a paragraph explaining a business strategy). In our multi‑agent graph, the supervisor routes questions containing tabular keywords (“year”, “table”, “trend”, “compare”) to the Genie node, while more qualitative queries go to the RAG node.

**2. How does the supervisor decide which node to route to?**  
The supervisor first inspects the step produced by the planner. Steps that start with “Retrieve:” are examined for tabular keywords (e.g., “year”, “table”, “trend”, “compare”, “highest”, “lowest”, “sql”, “tabular”). If any are present, the supervisor routes the step to the `genie_agent` node; otherwise it goes to the `rag_agent`. Steps that start with “Compute:” are sent to the `mcp_tools` node (which now uses UC‑governed functions). When all steps are completed, the supervisor routes to the `synthesizer`. This keyword‑based heuristic can be extended with LLM classification for higher accuracy.

**3. What are the tradeoffs of using Genie vs. RAG for structured data?**  
Genie provides precise, reproducible answers because it translates natural language into SQL and executes it on governed tables. However, it requires a well‑curated Delta table and a SQL warehouse (or a Genie Space). RAG is more flexible for mixed content – it can answer both numerical and narrative questions – but it may miss exact numerical relationships unless the data is explicitly present in the document chunks. Genie also adds governance benefits (permissions, lineage, audit), while RAG relies on the quality of the underlying Vector Search index.

## Part 3 – Agent Evaluation

**1. How did you measure the agent’s quality?**  
We built a small evaluation dataset of three question‑answer pairs covering retrieval, computation, and a mixed query. We used a simple substring‑match metric: the expected answer must appear inside the generated answer.

**2. What failure did you diagnose?**  
The mixed query “What was the revenue in 2023, and what would a 10% increase look like?” failed because the calculation step passed the literal phrase “revenue in 2023 * 1.10” to the `calculate` UC function, which could not parse it. The baseline graph produced an error message instead of the correct increased value.

**3. How did you fix it and prove the improvement?**  
We enhanced the `make_mcp_node` with variable substitution: before the LLM sees the step, financial terms like “revenue” are replaced with numeric values extracted from previous step results (e.g., 16.91 trillion → 16910000000000). After this fix, the same query returned the correct numeric result (18,601,000,000,000.0, i.e., ¥18.601 trillion).  
The substring‑match accuracy improved from **33.3% (baseline) to 66.7% (fixed)**: the mixed query now passes because the expected number (18.601 trillion) appears in the generated answer. The one remaining failure is a retrieval question where the RAG returned a slightly different net‑income figure (¥1,107 billion vs. the expected 1.124 billion). This is a data‑expectation mismatch, not a code bug. The critical functional failure (the mixed query) was resolved, demonstrating a genuine improvement in the agent’s reasoning capability.

**4. What are the limitations of this evaluation?**  
The dataset is tiny (3 questions) and the substring‑match metric is overly strict – it penalises correct answers expressed differently. A production evaluation would use a larger, more diverse dataset and LLM‑as‑judge metrics (e.g., `answer_correctness`, `faithfulness`) that can assess semantic equivalence rather than just lexical overlap.

### Part 4 – Challenge E: Guardrails

**How does a guardrail improve the safety and reliability of a deployed agent?**  
A guardrail acts as a safety net by catching malformed or nonsensical answers and replacing them with a benign fallback. This prevents the agent from displaying error messages or making unsupported claims, which maintains user trust and ensures consistent fallback behavior.


### Part 4 – Challenge F: Prompt Lifecycle

**What benefits does prompt versioning and aliasing bring to an agent in production?**  
Versioning prompts allows tracking changes over time, rolling back if a new prompt causes regressions, and comparing versions. Aliases decouple the prompt identity from its exact text, enabling updates without code changes. This supports safe, gradual rollouts and A/B testing.