# User-Friendly Q/A Workflow Report

## Search behavior

The interactive CLI now follows this workflow:

```text
search <topic>
    -> ranked arXiv candidates with deterministic scores
    -> highest-ranked candidate becomes the active paper
    -> fetch
```

The search output displays candidate titles, arXiv IDs, authors, publication
dates, abstracts, and ranking scores. It explicitly shows:

```text
Best match automatically selected: [1]
Next command:
fetch
Want another paper?
select <number>
```

The existing `select <number>` command changes the active paper to another
candidate from the latest search. Direct arXiv IDs and URLs remain supported.

## Q/A behavior

The active paper is persisted through the existing session store. Each
interactive `ask` command passes through the existing LangGraph workflow:

```text
question
    -> active paper
    -> Sentence Transformer embedding
    -> active paper's ChromaDB collection
    -> top five retrieved chunks
    -> grounding gate
    -> Gemini only when evidence is sufficient
    -> answer and sources
```

Conversation history is bounded and supplied only as additional context.
Every new question still performs a fresh vector retrieval.

## Unanswerable questions

Live verification produced:

```text
Paper: 1706.03762
Question: What is the population of India?
Result: QA_STATUS: INSUFFICIENT_EVIDENCE
Gemini: not used
```

The CLI now makes the refusal boundary explicit:

```text
Gemini was not used because the retrieved evidence was insufficient.
```

## Paper switching

The interactive `switch <number>` command continues to select an indexed
paper by number. It also supports an arXiv ID. After switching, the active
paper and subsequent retrieval collection change together.

Debug output confirms collection isolation, for example:

```text
Active paper: 1706.03762
Collection: paper_1706_03762
```

## Index reuse

The `fetch` command uses the existing index inspection path. A valid Chroma
collection reports `INDEX_STATUS: READY` and avoids downloading, parsing, and
embedding the paper again. The manual workflow exercised already indexed
papers and reported successful reuse.

## CLI usability

The following guidance was added:

- After `search`: best result, `fetch`, and `select <number>`.
- After `select`: selected paper and `fetch`.
- After `fetch`: `summarise`.
- After `summarise`: `ask <question>` with an example.
- After `ask`: another-question guidance.
- After `switch`: the new current paper and `ask <question>`.
- `help`: concise command descriptions and a typical workflow.

The search output no longer displays the removed low-level pipeline block:

```text
Next:
-> Fetch PDF
-> Parse PDF
-> Chunk
-> Generate embeddings
-> Store in ChromaDB
```

## GCN correction

```text
Wrong: 1606.09365
Correct: 1609.02907
Title: Semi-Supervised Classification with Graph Convolutional Networks
```

## Tests

```text
pytest -q
61 passed, 1 skipped

python -m compileall app tests
passed

git diff --check
passed
```

## Final status

| Requirement | Status |
| --- | --- |
| Ranked search results and exact existing scores | PASS |
| Automatic best-result selection | PASS |
| Numbered alternative selection | PASS |
| Active-paper Q/A without repeated paper IDs | PASS |
| Fresh retrieval for follow-up questions | PASS |
| Paper-specific Chroma isolation | PASS |
| Retrieved-evidence-only Gemini boundary | PASS |
| Insufficient-evidence Gemini refusal | PASS |
| Existing index reuse | PASS |
| Perfect semantic grounding calibration | PARTIAL |

The core architecture remains unchanged. No new database, model, agent, or
retrieval strategy was added.
