# Four-Minute Video Script — ArXiv Research Copilot

**Target length:** 3:45–4:00
**Recording style:** screen recording with voice-over
**Suggested demo paper:** `1706.03762` — *Attention Is All You Need*

## 0:00–0:20 — Opening

**On screen:** Show the repository README title, then switch to the terminal.

**Say:**

“Hello, I’m Amruth. This is ArXiv Research Copilot, a stateful research
assistant for discovering and understanding technical papers. The goal is not
to create another generic chatbot. The goal is to build a reliable research
loop that searches arXiv, reads one paper locally, creates a searchable
representation, and answers follow-up questions using evidence from that paper.
If the evidence is not there, the system says so.”

## 0:20–1:05 — Problem and architecture

**On screen:** Show the README architecture diagram or draw the flow:
`input → arXiv → PDF → chunks → embeddings → ChromaDB → grounding → Gemini`.

**Say:**

“The input can be a natural-language topic, an arXiv ID, or an arXiv URL. The
workflow is implemented as an explicit LangGraph state graph. The main stages
are query classification, arXiv retrieval, deterministic ranking or direct
paper resolution, indexing, retrieval, and response generation.

The shared state carries the user input, query type, candidate papers, selected
paper metadata, parsed text, chunks, collection name, retrieved evidence,
sources, briefing, and bounded conversation history. This makes the workflow
inspectable instead of hiding everything inside one large prompt.”

## 1:05–1:45 — Search and indexing demo

**On screen:** Run:

```bash
python -m app.main
```

Then enter:

```text
> search transformer architecture
```

**Say:**

“I’ll start an interactive session and search for transformer architecture.
Search uses the official arXiv API. The candidates are ranked using
deterministic title, abstract, and category signals, so the selection is
repeatable and easy to debug. The best match is activated automatically, but I
can select another result if needed.

Now I’ll fetch the active paper. The PDF is downloaded and parsed with PyMuPDF.
The text is split into page-aware, overlapping chunks. The chunks are embedded
locally with Sentence Transformers and stored in ChromaDB. Each paper has its
own deterministic collection, which prevents retrieval from mixing papers.”

**On screen:** Run:

```text
> fetch
```

Briefly point to the `PDF`, `Parsing`, `Chunks`, `Embeddings`, and `Index`
statuses.

## 1:45–2:25 — Executive briefing

**On screen:** Run:

```text
> summarise
```

**Say:**

“The briefing stage produces a structured executive summary. It includes the
paper title, authors, arXiv ID, publication information, why the paper matters,
the problem, approach, key claims, limitations, and suggested follow-up
questions. This is useful for deciding whether a paper deserves a deeper read,
without pretending that the briefing replaces the paper.”

Point at the limitations and follow-up questions in the output.

## 2:25–3:15 — Grounded QA demo

**On screen:** Run:

```text
> ask What is the main contribution?
> ask Why is multi-head attention useful?
```

**Say:**

“Now I’m entering QA mode. Every new question is embedded with the same local
model used for indexing. The retriever searches only the active paper’s
collection and returns the top relevant chunks. A grounding gate checks the
retrieval score before Gemini is called.

Gemini receives the question, a bounded amount of conversation history, and the
retrieved excerpts only. It does not receive the full PDF blindly. The response
also exposes page and section source metadata, so the answer can be inspected.”

If useful, run:

```text
> ask What is the population of India?
```

**Say:**

“This is an intentionally unrelated question. The system returns
`QA_STATUS: INSUFFICIENT_EVIDENCE` and does not call Gemini. That refusal path
is the core anti-hallucination behavior in this project.”

## 3:15–3:45 — Reliability and tradeoffs

**On screen:** Run:

```text
> status
> ask How does the method work? --debug
```

**Say:**

“The CLI also exposes operational status and retrieval diagnostics. Failures
are explicit: arXiv unavailability is different from a valid empty search,
PDF download is different from parsing, and parsing is different from local
indexing. If indexing fails after parsing, the paper remains retryable.

The main tradeoff is simplicity. I chose local embeddings and ChromaDB to avoid
an embedding service and external database. I chose deterministic ranking rather
than an LLM reranker for reproducibility. The known limitations are PDF layout,
tables and equations, the heuristic grounding threshold, and Gemini network
access.”

## 3:45–4:00 — Closing

**On screen:** Return to the README architecture and repository structure.

**Say:**

“In summary, ArXiv Research Copilot is a small but complete RAG system:
explicit state, official arXiv retrieval, local indexing, paper isolation,
grounded generation, citations, and a refusal path. The project is intentionally
CLI-first so the engineering focus stays on retrieval quality, state flow, and
reliable behavior. Thank you.”

## Recording checklist

- Start with the README hero and architecture diagram.
- Keep the terminal font large enough to read.
- Use one paper consistently during the demo.
- Show both a grounded answer and an insufficient-evidence refusal.
- Keep the final recording under four minutes.
- Do not display `.env`, API keys, or private credentials.
