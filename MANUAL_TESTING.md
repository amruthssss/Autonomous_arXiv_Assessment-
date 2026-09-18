# Manual testing checklist

Run from `D:\assessmen` with `.venv` active and `GEMINI_API_KEY` in `.env`.
MT-01 through MT-21 were previously validated before active-paper/removal
changes. Run affected tests plus MT-22 through MT-26 for the current build.

| ID | Test | Input | Expected result | Actual / status |
|---|---|---|---|---|
| MT-01 | Topic search | `python -m app.main search "graph neural networks"` | Candidates and scores print | |
| MT-02 | arXiv ID | `python -m app.main brief 1706.03762` | Exact paper resolves | |
| MT-03 | arXiv URL | `python -m app.main brief https://arxiv.org/abs/1706.03762` | Same paper resolves | |
| MT-04 | Candidate ranking | Topic with `-n 3` | Transparent scores are shown in JSON | |
| MT-05 | Selected paper | Brief a topic | One paper is selected | |
| MT-06 | PDF download | MT-02 first run | PDF appears under `data/pdfs` | |
| MT-07 | PDF parsing | MT-02 | Pages and text are extracted | |
| MT-08 | Extraction quality | Text-only paper | Poor extraction is reported | |
| MT-09 | Chunk creation | MT-02 | Chunks retain page/section metadata | |
| MT-10 | Embedding | MT-02 | Local model loads without HF token | |
| MT-11 | Chroma indexing | MT-02 | Persistent collection is created | |
| MT-12 | RAG retrieval | `ask ... --paper 1706.03762` | Relevant chunks are retrieved | |
| MT-13 | Executive briefing | `brief 1706.03762` | Structured briefing prints | |
| MT-14 | Limitations | Inspect briefing | Limitations are always present | |
| MT-15 | Known QA | Ask a paper-specific question | Answer has citations | |
| MT-16 | Follow-up QA | Use `python -m app.main prompt` | Follow-up uses same paper | |
| MT-17 | Unsupported QA | Capital of France with `--paper` | Exact refusal is returned | |
| MT-18 | Invalid input | `brief not-an-id` | Useful error, no traceback | |
| MT-19 | Zero results | Rare nonsense topic | Empty result is reported | |
| MT-20 | Cache/reuse | Repeat MT-02 | No duplicate PDF/index work | |
| MT-21 | Gemini/API failure | Temporarily use invalid key | Clear generation failure | |
| MT-22 | QA without `--paper` | Brief, then `ask "..."` | Active paper is reused | PASS — verified after persistent session implementation |
| MT-23 | Remove embeddings | `python -m app.main remove 1706.03762` | Only target collection removed; active cleared | PASS — verified; unrelated collections remained |
| MT-24 | Re-index after removal | Brief the removed paper | Cached PDF reused and Chroma rebuilt | PASS — verified; QA worked afterward |
| MT-25 | Missing active paper | Ask after removal | Clear no-active-paper message | PASS — verified |
| MT-26 | Cross-paper isolation | Query separate paper collections | Returned metadata belongs to queried paper | PASS — verified with paper-specific collections |

Automated validation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

For the optional live test:

```powershell
$env:RUN_NETWORK_TESTS="1"
.\.venv\Scripts\python.exe -m pytest -q tests/test_integration.py
```
