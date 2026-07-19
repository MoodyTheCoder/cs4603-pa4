# CS4603 PA4 — Document Analyst (Student Submission)

> This is your **submission file**. `README.md` is the assignment spec — this document is where you write up your work.
>
> - Document how to set up, run, and deploy your Document Analyst so a TA can reproduce your results.
> - **Answer every ANALYSIS QUESTION** from the assignment in the sections below.
> - Replace every `TODO` before submitting.
> - Keep it self-contained: a reader should be able to follow this file top-to-bottom —
>   setup → ingest → run → deploy → results — without opening the assignment spec.

## Setup

```bash
uv sync
cp .env.example .env   # then fill in your values
```

## Running locally


**Corpus ingestion (Task 0.3)**

`data/annual_report.pdf` was uploaded to a Unity Catalog volume at `/Volumes/main/default/pa4/annual_report.pdf` (22,167 bytes). From `pa4.ipynb`, I ran the ingestion pipeline, which:

1. Parsed the PDF with `ai_parse_document` into `main.default.pa4_parsed_documents`.
2. Chunked the parsed output into `main.default.pa4_chunks`.
3. Synced those chunks to the Vector Search endpoint `mehmood-vs-endpoint`, index `main.default.mehmood_analyst_index`, using managed embeddings (`databricks-gte-large-en`).

I confirmed the index was live by running a direct `similarity_search` against it before touching the graph — it returned the actual income-statement text from the report, including the FY2023 net revenue CAGR narrative and the condensed statement of operations table, so I knew retrieval was working before debugging anything upstream.

**Graph execution (Part 1)**

`agent.graph.build_graph()` wires together:

- **Planner** — turns the user's question into a JSON list of steps.
- **Supervisor** — looks at `current_step_index` against the plan and decides whether the next step goes to `rag_agent`, `mcp_tools`, or (once all steps are done) `synthesizer`.
- **RAG agent** — retrieves from the Vector Search index and writes a cited fact into `step_results`.
- **MCP tools node** — hands a compute step to an LLM that picks the right MCP tool (`calculate`, `growth_rate`, etc.) and calls it.
- **Synthesizer** — combines `step_results` into a final answer and appends it to `messages`.

I unit-tested each node individually before wiring the full graph:

| Node | Test input | Output |
|---|---|---|
| Planner | "What was net revenue in FY2023 and project 8% CAGR for 3 years?" | `['Retrieve: net revenue in FY2023', 'Compute: net revenue in FY2023 * (1 + 0.08)^3']` |
| Supervisor | step 0 of `['Retrieve: net revenue', 'Compute: 15% growth']` | `{'next_agent': 'rag_agent'}` — then `mcp_tools` at step 1, `synthesizer` at step 2 |
| RAG agent | "Retrieve: What was the net revenue in FY2023?" | "The net revenue in FY2023 was ¥16,910 billion." |
| MCP tools | "Compute: What is 15% of 2400?" | `0.15 * 2400 = 360` |
| Synthesizer | step result "Net revenue in FY2023 was ¥16.91 trillion" | "The revenue in FY2023 was ¥16.91 trillion... (source: annual_report.pdf, page 4)" |

Then I ran the full compiled graph on three queries:

| Query | Plan | Result |
|---|---|---|
| "What was the net income in 2020?" | `['Retrieve: net income in 2020']` | "The net income in 2020 was ¥455 billion, as found in Step 1." |
| "What is 15% of 2.4 billion?" | `['Compute: 2.4 billion * 0.15']` | Raw calc: `2.4e9 * 0.15 = 3.6e+08` → final answer: "360 million" |
| "What was the revenue in 2023, and what would a 10% increase look like?" | `['Retrieve: revenue in 2023', 'Compute: revenue in 2023 * 1.10']` | Step 1: ¥16.91 trillion (revenue). Step 2 used the MCP node's cross-step substitution to resolve "revenue in 2023" against the step 1 result and compute the 10% increase, producing a full combined answer with both figures. |

**Note:** this query worked as described above in earlier runs, but on the most recent run captured in the notebook, step 2 errored instead: `"Error evaluating 'revenue in 2023 * 1.10': Expression contains an unsupported operation"`, and the synthesizer fell back to reporting only the revenue figure. The substitution logic isn't consistently reliable — see Task 1.2 below for what I think is going on and why that matters for the architecture.

Offline smoke test (`pytest tests/test_smoke.py -q`) passes (2/2, with a couple of unrelated deprecation warnings from `databricks-vectorsearch` and Pydantic).

## Deployment

**Model definition (Task 2.1)** — `deployment/agent_model.py` validates required env vars at import time, builds the graph, and registers it with `mlflow.models.set_model(graph)`. Sanity check: `python -c "import deployment.agent_model"` → `✅ agent_model: graph loaded and set for MLflow.`

**Logging & registration (Task 2.2)** — `deployment/deploy.py` sets the MLflow tracking URI to Databricks and the registry to Unity Catalog, then logs the model with `mlflow.langchain.log_model` (code paths: `agent`, `rag`, `tools`, `config.py`). Running `log_and_register()` produced:

```
Registered model 'main.default.document_analyst' already exists. Creating a new version of this model...
Created version '13' of model 'main.default.document_analyst'.
```

(There's a dependency mismatch warning worth noting for future reproducibility: the logged model pins `langchain-mcp-adapters<0.0.5` while my environment had `0.3.0` installed. It didn't break anything here, but it's the kind of thing that could bite a TA re-running this later.)

**Serving endpoint (Task 2.3)** — the deploy script uses the Databricks SDK to create/update `pa4-doc-analyst-endpoint`. `DATABRICKS_HOST`, `DATABRICKS_TOKEN`, and `DATABRICKS_MODEL` are injected as endpoint secrets (`{{secrets/cs4603-deploy/...}}`); `VECTOR_SEARCH_ENDPOINT`, `VECTOR_SEARCH_INDEX`, and `EMBEDDINGS_ENDPOINT` are plain environment variables. Configured with `workload_size="Small"` and `scale_to_zero_enabled=True`. The endpoint reached `READY`:

```
✅ Endpoint is ready: https://dbc-b71811c7-c5ff.cloud.databricks.com/serving-endpoints/pa4-doc-analyst-endpoint/invocations
```

**Testing the endpoint (Task 2.4)** — a direct `requests.post` against the endpoint with `"What was the net income in 2023?"` returned:

```
✅ Final answer:
The net income in 2023 was ¥1,107 billion, as found in Step 1.
```

(This is profit attributable to owners for FY2023, taken straight from the condensed statement of operations the index returned. I originally had ¥1,124 billion written down for this figure, which is actually FY2023 *operating profit* — a different line on the same table. Worth double-checking exact line items when the report has this many nested subtotals.)

**Client SDK (Part 3, Task 3.2)** — using `client.sdk.DocumentAnalystClient`:

- `health_check()` returned `True`.
- `ask("What was the net income in 2023?")` returned the same ¥1,107 billion answer as the raw endpoint call.
- `ask_streaming("Summarize FY2023 revenue.")` streamed back a correct answer for revenue (¥16,910 billion, cited twice from two separate retrieval steps) but again hit a compute failure on a derived step ("total revenue minus total costs"), which the synthesizer reported honestly as `"Unable to process step"` rather than hallucinating a number.
- Timeout/retry behavior: a client configured with `timeout=0.001` and `max_retries=2` raised `TimeoutError: Request timed out after 0.001s and 3 attempts.` A client pointed at a nonexistent endpoint raised a `404 ENDPOINT_NOT_FOUND` error with the message text intact.


## Design decisions
- **Plan-and-execute over single-agent ReAct.** The planner decomposes the query up front, and the supervisor routes each step to a specialist. This gives a clear, inspectable trace (plan → step results → final answer) instead of one long ReAct scratchpad, which made debugging the failing compute step in Task 1.2 far easier than it would've been in a single-agent design.
- **Supervisor routing.** Routing is index-based (looks at `current_step_index` against the plan), not a full LLM classification per step, which keeps it fast and deterministic in testing.
- **MCP tools loaded once at graph build time**, wrapped synchronously via a background event loop so LangGraph nodes can call them without going async themselves.
- **Vector Search is queried live at inference time**, not baked into the model artifact — same retriever code runs locally and inside the serving container.
- **Synthesizer writes to both `final_answer` and `messages`.** This turned out to be load-bearing: MLflow serves the graph by reading the last entry in the `messages` channel, so if the synthesizer only set `final_answer`, the served endpoint would return nothing.

---

## Analysis Questions

> Answer in your own words. Each question is copied from the assignment so you don't have to flip back.

### Task 1.2 — Planner
1. What happens when the planner produces steps that depend on each other (e.g., step 3 needs the result of step 1)? How does your architecture handle this?
   The planner produces steps as plain text with no explicit data-passing between them — for the query "What was net revenue in FY2023 and project 8% CAGR for 3 years?", the planner literally output `'Compute: net revenue in FY2023 * (1 + 0.08)^3'` as step 2, with the phrase "net revenue in FY2023" left as a symbolic placeholder rather than a number. The MCP node has a keyword-matching heuristic that's supposed to scan prior `step_results` for a number when it sees a term like "revenue" and substitute it in before the expression reaches the calculator tool. For the combined query ("What was the revenue in 2023, and what would a 10% increase look like?"), this worked the way it was designed to in earlier runs: step 1 retrieved ¥16.91 trillion, and step 2's `'Compute: revenue in 2023 * 1.10'` had "revenue in 2023" substituted in correctly, giving a full combined answer.

   That said, it isn't fully reliable. On the most recent run of the exact same query, step 2 instead errored with `"Error evaluating 'revenue in 2023 * 1.10': Expression contains an unsupported operation"` — the substitution didn't happen that time, and the calculator tried to evaluate the raw text literally. The same category of failure showed up again in the streaming demo, where a derived "total revenue minus total costs" step came back as `"Unable to process step"`. So the heuristic is doing the right thing some of the time, but it's not deterministic enough to depend on — likely because keyword matching against `step_results` is brittle to small variations in how the RAG agent phrases its retrieved fact. A more robust fix would have the MCP node (or a small pre-processing LLM call) explicitly resolve step references into a fully numeric expression before it ever reaches the calculator tool, rather than relying on a keyword-match heuristic against the planner's natural-language phrasing.

2. Would a replanning step after each execution improve or hurt performance for this use case? Justify with an example.
   Given what I just found, replanning would probably help more than I originally thought. My first instinct was that financial QA plans are usually independent steps, so replanning would just add latency without benefit. But the combined-query failure shows a case where replanning could actually catch and fix a problem: after step 2 errors out with "unsupported operation," a replanning check could notice the failure, see that step 1's result (¥16.91 trillion) is sitting right there, and regenerate step 2 as a fully substituted expression like `16.91e12 * 1.10` instead of retrying the same broken string. Without replanning, the graph just reports the error and moves on. So for straightforward single-fact lookups replanning is overhead I don't need, but for any query where a later step consumes an earlier step's output, a lightweight replan-or-repair step after a failure would have fixed exactly the bug I hit.


### Task 1.3 — Supervisor
1. Your supervisor makes a routing decision per step. What is the failure mode if it misroutes? How would you detect and recover from a misroute?
   If a retrieval step got routed to `mcp_tools`, the calculator would choke on a natural-language question the same way it choked on "revenue in 2023 * 1.10" when substitution didn't happen — it can't evaluate text that isn't a numeric expression. If a compute step got routed to `rag_agent`, the retriever would search the vector index for something that isn't in the document and either return an irrelevant chunk or nothing useful. In my testing, the supervisor's index-based routing was correct in every case I ran (`rag_agent` → `mcp_tools` → `synthesizer` matched the plan exactly), so I didn't observe a live misroute in the supervisor itself — but the intermittent compute-step failure I did observe (working on one run, erroring on another for the identical query) produces the same downstream symptom: a `step_results` entry containing an error string that the synthesizer has to gracefully describe rather than silently pass off as a real answer. Detection could be built the same way I noticed the calculator error manually: check whether a step's result looks like an error/failure string, and if so, flag it for retry or re-routing before falling through to the synthesizer.

2. Compare this supervisor pattern with a single ReAct agent that has access to all tools. When is the supervisor pattern worth the added complexity?
   A single ReAct agent with both retrieval and calculator tools in one prompt is simpler to build, but it also means one model has to juggle two very different jobs — deciding when to search versus when to compute — inside a single reasoning loop, with no separation between "did I retrieve the right fact" and "did I compute correctly." The supervisor pattern splits those concerns into separate nodes with separate prompts, which is exactly what let me isolate the calculator failure to the MCP node specifically rather than debugging one tangled trace. It's worth the added complexity here because retrieval and calculation genuinely need different tuning (retrieval cares about chunk size and embedding quality; the calculator cares about getting a clean numeric expression), and because I want the plan itself as an auditable artifact, not just a hidden reasoning trace.


### Task 1.4 — RAG Agent
1. The RAG agent retrieves for a single decomposed step, not the full user query. How does this affect retrieval quality compared to retrieving for the original question?
   In my tests it helped. Retrieving on "Retrieve: What was the net revenue in FY2023?" instead of the full combined question ("What was the revenue in 2023, and what would a 10% increase look like?") gave the retriever a narrow, specific target, and it consistently returned the right figure (¥16,910 billion / ¥16.91 trillion) across every test I ran. If I'd sent the whole compound question to the retriever, the "10% increase" clause adds nothing for a vector search over an annual report and would likely just dilute the query embedding.

2. If the planner produces a vague step like "find relevant financial data," how would you improve the retrieval query before sending it to the vector store?
   I'd add a rewriting pass — either inside the RAG node or as a small step before it — that takes the vague planner step plus the original user question and asks the LLM to produce a specific, fact-shaped query (e.g., "What was FY2023 net revenue, operating profit, and net income?") before it hits the vector store. I didn't need this for my three test queries since the planner happened to produce specific steps each time, but the "revenue in 2023 * 1.10" substitution bug shows the planner's step text can't always be trusted to be precise enough to use downstream as-is, so the same rewriting logic would likely help both the RAG and MCP paths.

### Task 2.1 — Model Definition
1. Why does `models-from-code` require a self-contained file? What breaks if you reference external state (e.g., a database running only on your laptop)?
   The serving container Databricks spins up has no access to my laptop or any local process — it only has what's packaged via `code_paths` (in my case `agent`, `rag`, `tools`, `config.py`) plus whatever it can reach over the network. If `agent_model.py` referenced a local database, a file path on my machine, or anything not shipped in `code_paths`, the import would succeed locally but fail at model-load time inside the container, since none of that state exists there.

2. Your model calls a managed Vector Search index at inference time rather than embedding documents into the container image. What are the tradeoffs (freshness, cold-start size, latency, failure modes) of querying an external index vs. baking the corpus into the model artifact?
   - **Freshness:** querying the live index means document updates don't require a new model version; a baked-in corpus would go stale the moment the source document changes.
   - **Cold-start size:** keeping the corpus external kept the container lean — the model artifact only had to package code, not embeddings or raw chunks.
   - **Latency:** every retrieval call now costs a network round trip to Vector Search instead of an in-memory lookup, which is the tradeoff for the freshness/size wins.
   - **Failure modes:** this makes the whole system dependent on Vector Search being reachable. I actually saw an adjacent version of this risk in the client tests — pointing the client at a nonexistent endpoint returned a hard `404 ENDPOINT_NOT_FOUND`, which is the same category of failure a Vector Search outage would produce: a clean error rather than a silent wrong answer, at least.

### Task 2.3 — Serving Endpoint
1. Why must you pass `DATABRICKS_TOKEN` as an environment variable to the endpoint, even though it's already authenticated to serve models?
   The endpoint being "authenticated to serve models" only covers requests coming *into* the endpoint. Once my model code is running inside that container, it needs to make its own outbound calls to the LLM serving endpoint and the Vector Search index, and those calls need their own credentials — the endpoint's own service identity doesn't automatically inherit access to my workspace's data plane. That's why `DATABRICKS_TOKEN` gets injected as a secret rather than assumed.

2. What happens to in-flight requests when you deploy a new model version to the same endpoint? How does Databricks handle the transition?
   Databricks brings up new containers running the new model version (version 13, in my case) alongside the existing ones, waits for them to become healthy, and then shifts traffic over — the old containers keep serving whatever requests they already had in flight and are only torn down after those finish. So there's no dropped-request window during the version bump.


### Task 3.2 — Client
1. Why is exponential backoff better than fixed-interval retries for a model serving endpoint?
   If a bunch of clients are all retrying at the exact same fixed interval after an overload, they all hit the endpoint again at the same moment and can keep it overloaded. Backing off exponentially spreads those retries out over time instead of syncing them up, giving the endpoint room to recover.

2. Your client has a `max_retries` parameter. What is the danger of setting it too high in a production system with many concurrent users?
   I actually triggered this failure mode directly in testing: with `timeout=0.001` and `max_retries=2`, my client still made 3 total attempts before giving up (`Request timed out after 0.001s and 3 attempts.`). Scale that pattern up to many concurrent users with a high `max_retries`, and you get a retry storm — every failing client multiplies its own load on an already-struggling endpoint, ties up client-side connections/threads, and delays the point at which a genuine failure actually gets surfaced to whoever's watching.

3. When would you choose `ask_streaming()` over `ask()`? Give a concrete UX example.
   `ask_streaming()` is the right call whenever the answer is long enough that a user would rather start reading immediately than wait for the whole thing. My "Summarize FY2023 revenue" test is a decent example: the full answer was a multi-sentence summary combining two retrieval steps and a failed compute step. Streaming that in means the user sees the revenue figure appear almost immediately instead of staring at a blank response while the graph works through all three steps.

### Bonus A — CI/CD (if attempted)
1. Why should the deploy step only run on `main` and not on feature branches?
   - TODO
2. What would you add to this pipeline to prevent deploying a model that performs worse than the current version? Describe the gate.
   - TODO

### Bonus B — `databricks-agents` SDK (if attempted)
1. Compare the `agents.deploy()` approach with the manual MLflow + CLI approach from Part 2. What control do you gain or lose with each?
   - TODO
2. The Review App enables human feedback collection. How would you use this feedback to improve the agent over time? Describe a concrete feedback loop.
   - TODO

### Bonus C — Standalone MCP server (if attempted)
1. You moved the MCP server out of the model container. What did you gain (scaling, deployment, security, observability) and what new failure modes did you introduce (network, auth, latency, availability)?
   - TODO
2. The remote MCP server now needs its own authentication. How would you secure it so that only your serving endpoint — not the public internet — can call the tools?
   - TODO
3. When is bundling the tools in the container (Part 1) the *better* choice, and when is a separately deployed tool service (Bonus C) worth the extra moving parts?
   - TODO
