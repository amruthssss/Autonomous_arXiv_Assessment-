# Project Interview Knowledge Base

This document is an implementation-first guide to the repository at `D:\assessmen`.
It is intentionally based on the current source and tests, not on aspirational
product descriptions. Statements about functionality that is not present are
labelled **NOT IMPLEMENTED IN CURRENT REPOSITORY**. Claims in the README that do
not match the code are labelled **README VS IMPLEMENTATION DIFFERENCE**.

## 1. One-minute explanation

This is a Python CLI for researching arXiv papers with a local retrieval-
augmented generation (RAG) pipeline. A topic is sent to the official arXiv Atom
API, candidates are ranked deterministically, and the best paper is selected.
The PDF is cached, parsed page-by-page with PyMuPDF, normalized into overlapping
chunks, embedded locally with Sentence Transformers, and stored in one
persistent ChromaDB collection per paper. A question retrieves the top five
chunks from the active paper, applies a configurable best-score grounding gate,
and only then sends the excerpts (never the full PDF) to Gemini. Weak evidence
produces a refusal without a Gemini call. The CLI also persists the active
paper, caches arXiv responses and briefings, supports diagnostics/JSON output,
and can remove one paper's embeddings.

The workflow is deliberately linear:

```text
CLI input -> classify -> arXiv search/resolve -> deterministic rank
           -> download/parse/chunk -> embed/upsert -> retrieve
           -> grounding gate -> Gemini briefing/answer or refusal
```

## 2. Repository map

| Path | Responsibility |
|---|---|
| `app/main.py` | `argparse` one-shot commands, persistent interactive shell, rendering, export, active-paper handling |
| `app/graph.py` | Builds and compiles the LangGraph `StateGraph` |
| `app/nodes.py` | Query classification, scoring, search, indexing, retrieval, grounding, response generation |
| `app/state.py` | Pydantic domain models and `ResearchState` TypedDict |
| `app/services/config.py` | Environment loading and dependency construction |
| `app/services/arxiv.py` | Official API client, XML parsing, ID extraction, caching, retry/fallback/rate limiting |
| `app/services/pdf.py` | Safe PDF pathing/download, validation, extraction and chunking |
| `app/services/embeddings.py` | Lazy Sentence Transformer loading and normalized embeddings |
| `app/services/vector_store.py` | Persistent ChromaDB collections, upsert/query/inspection/removal |
| `app/services/gemini.py` | Gemini client, grounded prompts, structured briefing validation |
| `app/services/cache.py` | Atomic JSON cache keyed by SHA-256 of normalized keys |
| `app/services/session.py` | JSON persistence for the active paper only |
| `tests/` | Unit, node, graph, CLI, RAG, session and optional live integration tests |
| `data/` | Runtime cache/index/session files; PDFs and Chroma data are ignored by Git |

## 3. Architecture and execution traces

### Graph

`build_graph()` registers five nodes and fixed edges:

```text
START -> classify -> search -> index -> retrieve -> generate -> END
```

There are no conditional LangGraph edges. Nodes return partial dictionaries
which LangGraph merges into `ResearchState`; list fields use the `_replace`
reducer, so each node's value is authoritative rather than appended.

### Topic search trace

1. `classify_query()` honors an explicit classification or detects a direct
   arXiv ID/URL (`paper`), briefing words (`briefing`), question words/`?`
   (`qa`), otherwise `search`.
2. `search_papers()` clamps `max_results` to 1..50 and calls
   `ArxivService.search()`.
3. arXiv terms become `all:term AND all:term`; cached normalized `Paper` models
   are reused when available.
4. `rank_candidates()` computes title (55%), abstract (30%), and category (15%)
   overlap, with a phrase bonus capped at 1.0. Ties sort by publication string
   and arXiv ID. Only the first ranked paper is selected for indexing.
5. Search classification stops indexing/retrieval. The renderer shows either
   successful candidates, a valid `NO_RESULTS`, or a distinct API failure.

### Direct paper / briefing trace

1. `extract_arxiv_id()` normalizes versions away (`v2` is stored as the base
   ID), and `resolve()` performs an exact API lookup; it never broadens a direct
   paper request.
2. `index_papers()` checks the deterministic Chroma collection first. A valid
   collection is reused without downloading or parsing. A corrupted collection
   is deleted and rebuilt.
3. `PdfService.download()` reuses a valid cached PDF, otherwise downloads to a
   `.part` file, checks HTTP status/content type/size/PDF magic bytes, then
   atomically renames it.
4. `PdfService.extract()` uses PyMuPDF, retaining non-empty page numbers. It
   rejects invalid, empty, scanned/unextractable documents.
5. `chunks()` collapses whitespace, heuristically detects an uppercase section
   heading, and creates 1,200-character chunks with 180-character overlap and
   stable IDs of `paper_id:page:index`.
6. `ChromaVectorStore.add()` embeds chunks and upserts them into
   `paper_<sanitized_id>` with page, section, offsets and paper metadata.
7. Briefing retrieval uses `title + summary` as the query; QA uses the user's
   question. Retrieval is top five and filtered by `paper_id`.
8. `grounding_gate()` clamps scores to [0,1] and compares the best score with
   `GROUNDING_THRESHOLD` (default 0.20). A weak briefing returns a refusal;
   otherwise Gemini receives metadata and retrieved excerpts. Structured JSON is
   validated as `ExecutiveBriefing` and cached by paper ID plus query.

### Follow-up QA trace

The interactive shell retains `current` and the last five question/answer
turns. Every `ask` invokes the graph again, embeds the new question, retrieves
fresh chunks from the current paper collection, and applies the gate. History
is supplementary prompt context; it does not replace retrieval. Grounded
answers receive citations from the supplied excerpts. Refusal phrases returned
by Gemini are normalized to the same insufficient-evidence response.

## 4. State and data contracts

Important Pydantic models:

- `Paper`: arXiv metadata and URLs.
- `DocumentChunk`: text plus paper/page/section/start/end attribution.
- `SearchHit`: chunk and similarity-derived score (`1 - Chroma distance`).
- `SourceMetadata`: citation location.
- `CandidateScore`: explainable ranking components and matched terms.
- `ExecutiveBriefing`: title, metadata, summary, problem, findings, methods,
  results, limitations, follow-up questions and sources.

`ResearchState` includes query/classification/status fields, candidates and
selected papers, chunks/retrieved hits, errors, active collection, grounding
settings, history and debug flags. Statuses distinguish no results from
`arxiv_unavailable`, fetch/parse/index failures, and corrupted/not-indexed
collections.

Runtime data flow:

```text
arXiv XML -> Paper -> JSON cache
PDF bytes -> data/pdfs/<safe-id>.pdf -> page text -> DocumentChunk
chunks -> local embedding vectors -> data/chroma/paper_<id>
active paper -> data/session.json
briefing -> data/briefings.json
```

`.gitignore` excludes `.env`, virtualenv/cache files, PDFs, Chroma data,
`data/session.json`, and `data/*.json`; `.gitkeep` files preserve empty runtime
directories. Existing `data/arxiv.json`, `data/briefings.json`, and
`data/session.json` are local runtime artifacts, not source-of-truth fixtures.

## 5. Technology choices and alternatives

- **LangGraph:** explicit shared state and reproducible node boundaries.
  Alternatives: a plain pipeline, Prefect, or a web framework; a plain
  pipeline would be simpler for this fixed graph.
- **Official arXiv Atom API:** authoritative metadata and predictable XML.
  Alternatives: Semantic Scholar/Crossref, but they change coverage/semantics.
- **PyMuPDF:** fast local page text extraction. Alternatives include `pypdf`,
  Apache Tika, OCR (needed for scanned PDFs), or layout-aware parsers.
- **Sentence Transformers/all-MiniLM-L6-v2:** free, local, no per-query
  embedding API. Alternatives are hosted embeddings or a larger local model.
- **ChromaDB:** persistent local cosine search with one collection per paper.
  Alternatives include FAISS (less metadata/persistence), SQLite/pgvector,
  Elasticsearch or a managed vector database.
- **Gemini via `google-genai`:** synthesis and structured output after the
  local gate. Alternatives are another LLM or fully extractive answers.
- **Pydantic:** validates external/API/LLM data. Alternatives are dataclasses
  plus manual validation.
- **JSON caches and CLI:** low operational complexity for an assessment.
  A production service would need a database, concurrency controls, auth and
  observability.

## 6. Failures, edge cases and debugging

### Implemented handling

- Empty search results are not conflated with API/network failures.
- arXiv 406 tries the fallback endpoint once; 429/5xx and timeout/network
  errors retry with bounded exponential backoff and `Retry-After`, capped at
  30 seconds. Requests are paced by `ARXIV_MIN_REQUEST_INTERVAL`.
- Malformed XML, invalid IDs, missing PDF URLs, bad content type, oversized
  downloads (50 MB), invalid PDFs, zero-page PDFs and textless/scanned PDFs
  produce explicit errors.
- Failed indexing after successful parsing is retryable; valid indexes skip
  PDF work; corrupted indexes are removed before rebuild.
- Missing/malformed cache/session entries are treated as misses.
- A missing Gemini key, missing optional package, empty Gemini response or
  invalid briefing JSON becomes a generation error.
- Grounding blocks Gemini on low evidence; Gemini refusal language is also
  normalized.
- `remove` deletes only the deterministic target collection and clears the
  active session if it is that paper.

### Practical debugging

Use `--debug` on one-shot commands or `ask ... --debug` interactively. It
prints active paper, collection, index diagnostic, chunk IDs, scores, pages,
sections, grounding state and best score. Inspect `status`, `current`, and
`papers` in the interactive shell. Caches can be removed selectively under
`data/`; `remove <id>` safely removes one index. Unit tests inject openers,
services and fake vector stores, so network/LLM calls need not be used to
debug node behavior.

## 7. Testing and validation

The tests cover:

- deterministic classification, ranking, score components and graph wiring;
- arXiv parsing/cache keys, official headers/query construction, pacing,
  fallback, retry bounds, rate limits, timeout and network failures;
- PDF validation and stable chunk IDs;
- index reuse, retryable index failure, paper-isolated Chroma queries,
  repeated fresh retrieval and switching;
- Gemini prompt boundaries (retrieved chunks only), history forwarding,
  structured briefing requirements, cache reuse and no-call refusal behavior;
- CLI statuses/JSON/debug/source rendering and Markdown export;
- active-paper persistence and targeted embedding removal;
- an optional live pipeline guarded by `RUN_NETWORK_TESTS`.

Recommended checks from the repository:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app tests
git diff --check
```

The README and test reports record historical results of 63 passed and 1
skipped; run the commands above for the current checkout rather than relying
on that historical number.

## 8. Security, privacy, performance and scalability

### Security/privacy

Secrets are loaded from `.env` and ignored by Git; the API key is not embedded
in prompts or logs. PDF filenames are sanitized from arXiv IDs. Downloads use
timeouts, a size limit, content-type and magic-byte checks, and temporary
files. Gemini receives only selected excerpts, not the full paper. Chroma
collections and embeddings are local by default.

This is not a hardened multi-user service: there is no authentication,
authorization, encryption-at-rest policy, tenant isolation beyond paper IDs,
content scanning, prompt-injection defense for hostile PDF text, or remote
audit log. **NOT IMPLEMENTED IN CURRENT REPOSITORY:** user accounts, API
authorization, secret rotation, OCR malware sandboxing and production threat
model controls.

### Performance

ArXiv and briefing results are cached; PDFs and valid Chroma collections are
reused. Embedding model loading is lazy. Retrieval embeds one query and asks
for five chunks. Costs are still significant on first use: model download,
PDF parsing and embedding all chunks. Chunking is character-based, not token-
aware, and there is no batch/job queue.

### Scalability

One collection per paper is easy to isolate but collection enumeration and
JSON caches are local/process-oriented. `JsonCache` is “process-safe-enough”
but has no locking or eviction. Concurrent writers can conflict, and a single
machine's Chroma persistence is not a distributed architecture. **NOT
IMPLEMENTED IN CURRENT REPOSITORY:** web deployment, horizontal workers,
distributed vector storage, background ingestion, quotas, monitoring,
multi-user sessions or streaming responses.

## 9. Limitations and extension ideas

Current limitations include heuristic section detection, character chunks,
imperfect tables/equations/layout extraction, no OCR, uncalibrated similarity
threshold, no reranker, possible false grounding, dependency/model downloads,
network/API/Gemini availability, and no citation verification beyond prompting
and source metadata.

Reasonable extensions:

1. Add OCR/layout-aware extraction and table/equation preservation.
2. Make chunk size/overlap token-aware and evaluate retrieval with labelled
   questions; calibrate the threshold per embedding model.
3. Add a cross-encoder reranker and citation entailment/quote verification.
4. Version indexes by embedding model and parser settings; add cache TTL and
   file locks.
5. Add resumable ingestion, progress reporting, structured logs and metrics.
6. Move metadata/cache/indexes to a transactional or managed backend for
   multi-user deployment, with authentication and tenant boundaries.
7. Add explicit prompt-injection handling for untrusted paper text.

## 10. README versus implementation

- **README VS IMPLEMENTATION DIFFERENCE:** the README's “Shared state” table
  names `user_input`, `query_type`, `candidates`, `selected_paper`,
  `parsed_text`, and `retrieved_chunks`; the code uses `query`,
  `classification`, `papers`, `selected_papers`, `chunks`, and `retrieved`.
  The diagrams are conceptual, not literal field names.
- **README VS IMPLEMENTATION DIFFERENCE:** README example text says a direct
  paper may be entered at the prompt and describes “Open a specific paper” as
  typing an ID; the persistent shell supports this, while one-shot direct
  paper behavior is through `brief/ask --paper` or `prompt`.
- **README VS IMPLEMENTATION DIFFERENCE:** README describes “latest
  validation” as 63 passed/1 skipped, which is a historical report, not a
  guaranteed current result.
- The README's core claims—paper-isolated retrieval, five-hit QA, local
  embeddings, grounding gate, active paper, retries and refusal behavior—are
  implemented and covered by tests.

## 11. Interview questions and answers

### Beginner

**Q1. What is RAG here?**
Retrieval-augmented generation: retrieve relevant paper chunks first, then
give only those excerpts to Gemini to synthesize an answer.

**Q2. Why is Chroma split by paper?**
`collection_name()` deterministically creates `paper_<id>`, and queries also
filter `paper_id`; this prevents answers from mixing indexed papers.

**Q3. What happens on the first fetch?**
The PDF is downloaded and validated, text is extracted and chunked, the local
embedding model loads, vectors are upserted, and the collection persists.

**Q4. How does a user run the application?**
Install `requirements.txt`, configure `.env`, then run `python -m app.main`.
The shell supports `search`, `select`, `fetch`, `summarise`, `ask`, `papers`,
`switch`, `status`, `current`, `export`, `help`, and `exit`.

**Q5. What does `.gitignore` protect?**
Environment secrets, virtualenv/cache files, downloaded PDFs, Chroma data,
JSON runtime state and generated caches.

### Intermediate

**Q6. How is a topic ranked?**
Terms are normalized and de-duplicated; title, abstract and category term
overlap are weighted 0.55/0.30/0.15, with a phrase bonus, then deterministic
tie-breakers are applied.

**Q7. How does the system avoid hallucinating on unrelated questions?**
The best retrieved similarity score must meet the clamped threshold (default
0.20). Otherwise the node returns insufficient evidence and does not call
Gemini.

**Q8. How are transient arXiv failures handled?**
406 switches once to the fallback official endpoint; 429/5xx/timeouts/network
errors use bounded retries, exponential backoff and optional `Retry-After`.

**Q9. How is index reuse detected?**
`inspect_index()` checks collection existence, IDs, documents, metadata,
embeddings, lengths and matching `paper_id`. Valid collections are reused;
corrupt ones are deleted and rebuilt.

**Q10. Why pass conversation history if retrieval is repeated?**
History supports follow-up pronouns/context for Gemini, while fresh retrieval
keeps every answer grounded in the current question and active paper.

### Advanced

**Q11. Is the grounding score a probability?**
No. It is `1 - Chroma cosine distance`, clamped to [0,1], and the maximum of
five hits. It is a heuristic; it is not calibrated or entailment-checked.

**Q12. What is the security boundary around Gemini?**
The gate and prompt boundary restrict Gemini to retrieved excerpts and bounded
history. However, paper text is untrusted input and there is no dedicated
prompt-injection detector or citation verifier.

**Q13. What consistency risks exist in the caches?**
JSON writes are atomic via a temporary sibling file, but there are no locks,
TTL, schema/version migration or multi-process conflict resolution. A malformed
entry becomes a cache miss.

**Q14. How would you scale ingestion?**
Separate search/ingestion/QA services, use a job queue and durable metadata
store, version collections by parser/embedding model, use a managed vector
database, add per-tenant authorization, and expose metrics/retries. The
current synchronous CLI is not that architecture.

**Q15. What would you test before changing the retrieval model?**
Golden questions with expected supporting pages, recall/precision and refusal
metrics, threshold calibration, cross-paper isolation, index-version
compatibility, prompt evidence boundaries, and regression tests for no-Gemini
calls below threshold.

**Q16. Why might a valid paper still fail?**
The PDF can be scanned or layout-heavy, arXiv/Gemini can be unavailable, the
embedding package/model may be missing, Chroma can be corrupt, or the query
may retrieve below the heuristic threshold.

**Q17. What does the graph not do?**
It has no branching edges, streaming, autonomous tool loop, evaluator node,
multi-paper synthesis, or web API. Those are **NOT IMPLEMENTED IN CURRENT
REPOSITORY**; classification affects node behavior inside the fixed sequence.

**Q18. How would you explain the most important tradeoff?**
Local embeddings and paper-isolated persistence improve privacy, cost and
reproducibility, but first-run latency, model quality and local storage
management are worse than a hosted production retrieval stack.
