# Follow-up Q/A RAG Verification Report

## Q/A Architecture

The interactive `ask` command keeps the active paper ID from the session and
passes each new question through the existing LangGraph workflow:

```text
User question
    -> active paper ID
    -> local Sentence Transformer embedding
    -> paper-specific ChromaDB query (top 5)
    -> best-score grounding gate
    -> Gemini with retrieved chunks and bounded history only
    -> answer and source metadata
```

`ChromaVectorStore.query()` uses both the deterministic paper collection and
`where={"paper_id": paper_id}`. Conversation history is passed as additional
context to Gemini, but `retrieve_context()` is executed again for every
question.

If the best retrieved score is below `GROUNDING_THRESHOLD` (default `0.20`),
the QA node returns `QA_STATUS: INSUFFICIENT_EVIDENCE` and does not call
Gemini.

## Manual Questions Tested

The existing local indexes were reused. Each QA command retrieved five chunks.

| Paper | Question | Retrieval | Grounded | Gemini Called | Result |
| --- | --- | --- | --- | --- | --- |
| 1706.03762 | What is the main contribution of this paper? | Yes, `paper_1706_03762` | Yes, 0.2628 | Yes | PASS |
| 1706.03762 | What architecture does the paper propose? | Yes, `paper_1706_03762` | Yes, 0.2541 | Yes | PASS |
| 1706.03762 | How does self-attention work in the proposed architecture? | Yes, `paper_1706_03762` | Yes, 0.5551 | Yes | PASS |
| 1706.03762 | Why does the paper use multi-head attention? | Yes, `paper_1706_03762` | Yes, 0.5548 | Yes | PASS |
| 1706.03762 | What are the limitations mentioned by the authors? | Yes, `paper_1706_03762` | Yes, 0.2281 | Yes | PASS |
| 1706.03762 | What is the population of India? | Yes, `paper_1706_03762` | No, 0.0665 | No | PASS |
| 2005.11401 | What is the main idea of retrieval-augmented generation? | Yes, `paper_2005_11401` | Yes, 0.5726 | Yes | PASS |
| 2005.11401 | What problem is the method trying to solve? | Yes, `paper_2005_11401` | No, 0.1732 | No | PARTIAL: evidence below threshold |
| 2005.11401 | What are the main components of the proposed approach? | Yes, `paper_2005_11401` | Yes, 0.2373 | Yes | PASS |
| 2005.11401 | What is the Transformer architecture? | Yes, `paper_2005_11401` | Yes, 0.3278 | Yes | PARTIAL: semantic overlap is heuristic |
| 2005.11401 | What is the population of India? | Yes, `paper_2005_11401` | No, 0.1609 | No | PASS |
| 1609.02907 | What is the main idea of graph convolution? | Yes, `paper_1609_02907` | Yes, 0.5639 | Yes | PASS |
| 1609.02907 | What are the limitations of the proposed approach? | Yes, `paper_1609_02907` | No, 0.1584 | No | PARTIAL: evidence below threshold |

## Retrieval Verification

Debug output confirmed the active collection and five retrieved chunks for
each question. Returned hits preserve:

- `paper_id`
- `chunk_id`
- `page`
- `section`
- chunk text
- similarity score

Switching papers changed the collection deterministically:

```text
1706.03762 -> paper_1706_03762
2005.11401 -> paper_2005_11401
1609.02907 -> paper_1609_02907
```

No cross-paper collection was used.

## Grounding Verification

Grounding uses the existing best similarity score and threshold `0.20`.
Successful questions produced `QA_STATUS: GROUNDED`. Unrelated India-population
questions produced `QA_STATUS: INSUFFICIENT_EVIDENCE`.

The threshold remains unchanged. Similarity is documented as a practical
heuristic rather than a calibrated probability.

## Gemini Verification

Automated tests prove that grounded QA passes only retrieved `SearchHit`
objects to Gemini and excludes full parsed PDF text. They also prove that a
low-score QA response returns the refusal text while the Gemini mock call
count remains zero.

The prompt includes the user question, bounded prior turns when available, and
retrieved excerpts formatted with arXiv/page/section provenance.

## Unanswerable Question

```text
Paper: 2005.11401
Question: What is the population of India?
Grounding score: 0.1609
Expected: QA_STATUS: INSUFFICIENT_EVIDENCE
Actual: QA_STATUS: INSUFFICIENT_EVIDENCE
Gemini called: NO
```

The same question on `1706.03762` scored `0.0665` and was also refused.

## Follow-up Context

The following sequence was tested on `1706.03762`:

```text
ask What architecture does the paper propose?
ask Why is it useful?
```

Both turns retrieved five fresh chunks from `paper_1706_03762`. The second
turn also received bounded conversation history as additional Gemini context;
history did not replace retrieval.

## Paper Switching

The sequence below was tested:

```text
switch 1706.03762
ask What is the main contribution?
switch 2005.11401
ask What is the main contribution?
switch 1706.03762
ask What is the main contribution?
```

Each answer queried the collection belonging to the active paper. Chroma
metadata filtering provides a second paper-ID isolation check.

## Index Reuse

The manual run reported `FETCH_STATUS: SUCCESS` and `INDEX_STATUS: READY`
while using existing indexes. The index inspection path reports that existing
collections are reused and avoids PDF download, parsing, and re-embedding.
Focused automated coverage verifies that a valid index prevents PDF download
and parsing.

## GCN Correction

```text
Wrong: 1606.09365
Correct: 1609.02907
Title: Semi-Supervised Classification with Graph Convolutional Networks
```

The corrected paper was used for the manual GCN verification.

## Automated Tests

```text
pytest -q
61 passed, 1 skipped

python -m compileall app tests
passed

git diff --check
passed
```

Focused tests cover fresh retrieval, active-paper isolation, repeated
questions, paper switching, index reuse, Gemini evidence boundaries, and
no-Gemini insufficient-evidence behavior.

## Requested Presentation Change

Removed the topic-search output block:

```text
Next:
-> Fetch PDF
-> Parse PDF
-> Chunk
-> Generate embeddings
-> Store in ChromaDB
```

The output test now asserts that this block is absent.

## Final Result

| Requirement | Result |
| --- | --- |
| Every QA question retrieves from ChromaDB | PASS |
| Active paper controls retrieval | PASS |
| Follow-up questions perform fresh retrieval | PASS |
| Gemini receives retrieved evidence only | PASS |
| Insufficient evidence blocks Gemini | PASS |
| Source/page metadata is preserved | PASS |
| Existing indexes are reused | PASS |
| Grounding quality is perfectly calibrated for every semantic question | PARTIAL |

The core Q/A RAG behavior passes. The remaining limitation is the intentionally
simple best-similarity grounding heuristic; no additional reranker, classifier,
LLM judge, or retrieval architecture was introduced.
