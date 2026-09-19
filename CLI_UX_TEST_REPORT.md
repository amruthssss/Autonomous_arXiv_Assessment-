# CLI UX Test Report

## What changed

- Search results now show only title, arXiv ID, and deterministic relevance.
- The highest-ranked result remains automatically selected.
- Fetch output uses clear PDF, parsing, chunks, embeddings, and index statuses.
- A failed local embedding/index attempt is explicitly retryable with `fetch`.
- Normal source output shows page/section only; chunk IDs remain available in debug output.
- Gemini refusal text is converted to `QA_STATUS: INSUFFICIENT_EVIDENCE`.
- Startup, direct-paper selection, and next-command guidance were simplified.

## Before vs after fetch failure

Before:

```text
EMBEDDINGS: FAILED
INDEX: FAILED
ERROR: Embedding model failed during initialization.
Please run fetch again.
```

After:

```text
PDF          [OK] Ready
Parsing      [OK] Success
Chunks       [OK] 73
Embeddings   [FAILED] Not ready
Index        [FAILED] Not ready

The paper was downloaded and parsed successfully, but local indexing could not
be completed. Run fetch again to retry.

Next command:
fetch
```

Successful retry continues to show:

```text
Embeddings   [OK] Ready
Index        [OK] Ready

Paper is ready for research.

Next command:
summarise
```

## Search test

`search transformer architecture` displayed ranked candidates with exact
existing scores, without full abstracts, URLs, or component-score noise. The
highest deterministic score was automatically made active and the CLI showed
`fetch` as the next command.

## Selection test

`select 2` changes the active paper from the latest search results. Already
indexed papers report `INDEX_STATUS: READY` and guide the user to
`summarise` rather than requiring a new fetch.

## Fetch test

`fetch` preserves PDF, parsing, chunk, embedding, and index status reporting.
Valid existing Chroma collections are reused. A vector-store failure after
successful parsing is classified as retryable.

## Summarise test

`summarise` continues to use the selected active paper and the existing
retrieved evidence pipeline.

## QA and follow-up tests

Interactive questions continue to use the active paper automatically. Each
question performs fresh embedding and Chroma retrieval before the grounding
gate and Gemini call. Conversation history remains supplementary context.

## Insufficient-evidence test

The unrelated population question is rejected by the grounding gate when
evidence is insufficient:

```text
QA_STATUS: INSUFFICIENT_EVIDENCE
Gemini was not used because the retrieved evidence was insufficient.
```

If Gemini itself returns an evidence-refusal phrase after passing the heuristic
gate, the final result is also normalized to this same refusal status.

## Paper switching and persistence

`papers`, `switch <number>`, `current`, and the active-paper session store
remain supported. Switching changes the paper-specific Chroma collection
without re-downloading or re-embedding a valid existing index.

## Direct arXiv ID test

Entering an arXiv ID or URL directly remains supported. The CLI reports
whether the paper is already indexed and recommends `summarise` or `fetch`.

## Runtime cleanup

Disposable cached PDFs and generated Chroma files were removed to reclaim
workspace space. The required `.gitkeep` files remain. Source code, tests,
configuration, documentation, and runtime metadata files were preserved.

## Test results

```text
pytest -q: 63 passed, 1 skipped
python -m compileall app tests: passed
git diff --check: passed
```

## Remaining issue

The grounding score remains the intentionally simple best-retrieved-chunk
heuristic. No additional reranker, model, agent, or database was introduced.
