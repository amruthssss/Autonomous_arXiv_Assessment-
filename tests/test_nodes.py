from app.nodes import (
    classify_query,
    generate_response,
    grounding_gate,
    index_papers,
    rank_candidates,
    retrieve_context,
    search_papers,
)
from app.services.arxiv import ArxivAPIError
from app.state import DocumentChunk, ExecutiveBriefing, Paper, SearchHit


def test_classification_is_deterministic():
    assert classify_query({"query": "compare transformer papers"})["classification"] == "briefing"
    assert classify_query({"query": "How does retrieval work?"})["classification"] == "qa"
    assert classify_query({"query": "neural retrieval"})["classification"] == "search"
    direct = classify_query({"query": "https://arxiv.org/abs/2301.12345v2"})
    assert direct["classification"] == "paper"
    assert direct["direct_paper_id"] == "2301.12345"


def test_candidate_score_exposes_relevance_components():
    papers = [
        Paper(arxiv_id="1", title="Neural retrieval", summary="dense retrieval", categories=["cs.IR"]),
        Paper(arxiv_id="2", title="Robotics", summary="control", categories=["cs.RO"]),
    ]
    ranked = rank_candidates(papers, "neural retrieval")
    assert ranked[0][0].arxiv_id == "1"
    assert ranked[0][1].title_relevance > ranked[1][1].title_relevance
    assert ranked[0][1].total >= ranked[1][1].total


def test_ranking_is_deterministic_and_selects_highest_candidate():
    papers = [
        Paper(arxiv_id="1", title="Graph neural networks", summary="graph neural networks", categories=["cs.LG"]),
        Paper(arxiv_id="2", title="Unrelated systems", summary="systems", categories=["cs.OS"]),
    ]
    first = search_papers(
        {"query": "graph neural networks", "classification": "search", "max_results": 5},
        type("Services", (), {"arxiv": type("Arxiv", (), {"search": lambda *_args, **_kwargs: papers})()})(),
    )
    second = search_papers(
        {"query": "graph neural networks", "classification": "search", "max_results": 5},
        type("Services", (), {"arxiv": type("Arxiv", (), {"search": lambda *_args, **_kwargs: papers})()})(),
    )
    assert [score.total for score in first["candidate_scores"]] == [
        score.total for score in second["candidate_scores"]
    ]
    assert first["selected_papers"][0].arxiv_id == first["papers"][0].arxiv_id
    assert first["search_status"] == "success"


def test_empty_search_has_distinct_no_results_status():
    result = search_papers(
        {"query": "no matches", "classification": "search", "max_results": 5},
        type(
            "Services",
            (),
            {"arxiv": type("Arxiv", (), {"search": lambda *_args, **_kwargs: []})()},
        )(),
    )
    assert result["papers"] == []
    assert result["search_status"] == "no_results"


def test_arxiv_failure_is_not_reported_as_no_results():
    def fail(*_args, **_kwargs):
        raise ArxivAPIError.from_http(406)

    result = search_papers(
        {"query": "blocked topic", "classification": "search", "max_results": 5},
        type("Services", (), {"arxiv": type("Arxiv", (), {"search": fail})()})(),
    )
    assert result["search_status"] == "arxiv_unavailable"
    assert result["papers"] == []


def test_unresolved_paper_does_not_become_qa_insufficient_evidence():
    result = generate_response(
        {
            "classification": "qa",
            "search_status": "no_results",
            "paper_status": "not_found",
            "errors": ["Could not resolve arXiv paper 9999.99999"],
        },
        type("Services", (), {})(),
    )
    assert result["answer"] is None
    assert result["errors"] == ["Could not resolve arXiv paper 9999.99999"]


def test_briefing_retrieval_uses_paper_content_not_identifier():
    captured = {}
    paper = Paper(
        arxiv_id="1706.03762",
        title="Attention Is All You Need",
        summary="We propose a new simple network architecture based on attention.",
    )
    vector = type(
        "Vector",
        (),
        {
            "query": lambda self, text, paper_id, n_results=5: (
                captured.update(text=text) or []
            )
        },
    )()
    services = type("Services", (), {"vector": vector})()
    retrieve_context(
        {
            "query": "1706.03762",
            "classification": "briefing",
            "selected_papers": [paper],
            "paper_index_status": "indexed",
        },
        services,
    )
    assert captured["text"] == (
        "Attention Is All You Need. "
        "We propose a new simple network architecture based on attention."
    )


def test_search_failure_does_not_download_or_index():
    class Pdf:
        def download(self, _paper):
            raise AssertionError("PDF must not be downloaded after search failure")

    result = index_papers(
        {
            "classification": "search",
            "search_status": "error",
            "errors": ["HTTP 406"],
        },
        type("Services", (), {"pdf": Pdf()})(),
    )
    assert result["paper_index_status"] == "not_checked"
    assert result["errors"] == ["HTTP 406"]


def test_valid_existing_index_skips_pdf_download():
    paper = Paper(arxiv_id="1706.03762", title="Attention Is All You Need")

    class Pdf:
        def download(self, _paper):
            raise AssertionError("PDF must not be downloaded for an indexed paper")

        def chunks(self, _paper):
            raise AssertionError("PDF must not be parsed for an indexed paper")

    class Vector:
        def inspect_index(self, paper_id):
            assert paper_id == paper.arxiv_id
            return "indexed", "Paper already indexed locally."

        def collection_name(self, paper_id):
            return f"paper_{paper_id.replace('.', '_')}"

    result = index_papers(
        {"classification": "briefing", "search_status": "success", "selected_papers": [paper]},
        type("Services", (), {"pdf": Pdf(), "vector": Vector()})(),
    )
    assert result["paper_index_status"] == "indexed"
    assert result["chunks"] == []
    assert "already indexed" in result["index_diagnostic"]


def test_index_failure_after_parse_is_retryable_and_descriptive():
    paper = Paper(arxiv_id="1", title="Paper")
    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="Paper", text="text", page=1)

    class Pdf:
        def download(self, _paper):
            return None

        def chunks(self, _paper):
            return [chunk]

    class Vector:
        def inspect_index(self, _paper_id):
            return "not_indexed", "No index"

        def add(self, _chunks):
            raise RuntimeError("embedding model failed")

        def collection_name(self, paper_id):
            return f"paper_{paper_id}"

    result = index_papers(
        {"classification": "briefing", "search_status": "success", "selected_papers": [paper]},
        type("Services", (), {"pdf": Pdf(), "vector": Vector()})(),
    )

    assert result["fetch_status"] == "success"
    assert result["parse_status"] == "success"
    assert result["index_status"] == "failed"
    assert "downloaded and parsed successfully" in result["index_diagnostic"]
    assert "Run fetch again" in result["index_diagnostic"]


def test_grounding_gate_refuses_low_scores():
    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="T", text="evidence", page=2)
    assert grounding_gate([SearchHit(chunk=chunk, score=0.2)], 0.35) == (False, 0.2)
    assert grounding_gate([SearchHit(chunk=chunk, score=0.8)], 0.35) == (True, 0.8)


def test_qa_passes_only_retrieved_chunks_to_gemini():
    retrieved = DocumentChunk(
        chunk_id="1",
        paper_id="1",
        title="Paper",
        text="The retrieved evidence.",
        page=2,
    )
    full_pdf_text = "FULL PDF TEXT THAT MUST NEVER REACH GEMINI"

    class Gemini:
        def __init__(self):
            self.calls = 0
            self.hits = []

        def answer(self, query, hits, history=None):
            self.calls += 1
            self.hits = hits
            return "Grounded answer."

    gemini = Gemini()
    result = generate_response(
        {
            "classification": "qa",
            "query": "What is the contribution?",
            "retrieved": [SearchHit(chunk=retrieved, score=0.8)],
            "parsed_text": full_pdf_text,
            "grounding_threshold": 0.2,
        },
        type("Services", (), {"gemini": gemini, "grounding_threshold": 0.2})(),
    )

    assert result["answer"] == "Grounded answer."
    assert gemini.calls == 1
    prompt_input = "\n".join(hit.chunk.text for hit in gemini.hits)
    assert retrieved.text in prompt_input
    assert full_pdf_text not in prompt_input


def test_low_score_qa_refuses_without_calling_gemini():
    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="Paper", text="weak", page=1)

    class Gemini:
        calls = 0

        def answer(self, *args, **kwargs):
            self.calls += 1
            return "Should not be generated."

    gemini = Gemini()
    result = generate_response(
        {
            "classification": "qa",
            "query": "Unsupported question",
            "retrieved": [SearchHit(chunk=chunk, score=0.19)],
            "grounding_threshold": 0.2,
        },
        type("Services", (), {"gemini": gemini, "grounding_threshold": 0.2})(),
    )

    assert gemini.calls == 0
    assert result["answer"] == "I couldn't find enough information in the paper to answer that."


def test_briefing_requires_limitations_and_follow_ups():
    required = ExecutiveBriefing.model_fields
    assert required["limitations"].is_required()
    assert required["follow_up_questions"].is_required()


def test_briefing_cache_reuses_structured_result():
    paper = Paper(arxiv_id="1", title="Cached")

    class Cache:
        def __init__(self):
            self.values = {}

        def get(self, key):
            return self.values.get(key)

        def set(self, key, value):
            self.values[key] = value

    class Gemini:
        calls = 0

        def briefing(self, papers, hits, query):
            self.calls += 1
            return ExecutiveBriefing(
                title="Cached",
                arxiv_id="1",
                executive_summary="Summary",
                problem_statement="Problem",
                key_findings=["Finding"],
                methods="Method",
                results=["Result"],
                limitations=["No explicit limitation identified."],
                follow_up_questions=["Question"],
                sources=[],
            )

    cache = Cache()
    gemini = Gemini()
    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="Cached", text="evidence", page=1)
    services = type(
        "Services",
        (),
        {"briefing_cache": cache, "gemini": gemini, "grounding_threshold": 0.2},
    )()
    state = {
        "classification": "briefing",
        "selected_papers": [paper],
        "query": "brief",
        "retrieved": [SearchHit(chunk=chunk, score=0.8)],
    }

    generate_response(state, services)
    result = generate_response(state, services)

    assert result["briefing"].title == "Cached"
    assert gemini.calls == 1


def test_briefing_refuses_without_grounded_evidence():
    class Gemini:
        def briefing(self, *_args, **_kwargs):
            raise AssertionError("Gemini must not be called")

    result = generate_response(
        {
            "classification": "briefing",
            "selected_papers": [Paper(arxiv_id="1", title="Paper")],
            "query": "brief",
            "retrieved": [],
        },
        type("Services", (), {"gemini": Gemini(), "grounding_threshold": 0.2})(),
    )
    assert result["grounded"] is False
    assert "couldn't find enough information" in result["briefing"]


def test_briefing_refuses_low_scoring_evidence():
    class Gemini:
        def briefing(self, *_args, **_kwargs):
            raise AssertionError("Gemini must not be called")

    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="Paper", text="weak", page=1)
    result = generate_response(
        {
            "classification": "briefing",
            "selected_papers": [Paper(arxiv_id="1", title="Paper")],
            "query": "brief",
            "retrieved": [SearchHit(chunk=chunk, score=0.19)],
        },
        type("Services", (), {"gemini": Gemini(), "grounding_threshold": 0.2})(),
    )
    assert result["grounded"] is False
    assert result["grounding_score"] == 0.19


def test_qa_forwards_conversation_history_to_gemini():
    received = {}

    class Gemini:
        def answer(self, query, hits, history=None):
            received["history"] = history
            return "answer"

    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="Paper", text="evidence", page=1)
    history = [{"question": "What is the method?", "answer": "It uses retrieval."}]
    result = generate_response(
        {
            "classification": "qa",
            "query": "What are its limitations?",
            "retrieved": [SearchHit(chunk=chunk, score=0.8)],
            "conversation_history": history,
        },
        type("Services", (), {"gemini": Gemini(), "grounding_threshold": 0.2})(),
    )
    assert result["answer"] == "answer"
    assert received["history"] == history


def test_qa_converts_gemini_insufficient_evidence_to_refusal():
    class Gemini:
        def answer(self, *_args, **_kwargs):
            return "I couldn't find enough information in the provided excerpts to answer this."

    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="Paper", text="evidence", page=1)
    result = generate_response(
        {
            "classification": "qa",
            "query": "What is the architecture?",
            "retrieved": [SearchHit(chunk=chunk, score=0.8)],
        },
        type("Services", (), {"gemini": Gemini(), "grounding_threshold": 0.2})(),
    )

    assert result["grounded"] is False
    assert result["answer"] == "I couldn't find enough information in this paper to answer that."
