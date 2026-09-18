from app.nodes import classify_query, generate_response, grounding_gate, rank_candidates, search_papers
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


def test_grounding_gate_refuses_low_scores():
    chunk = DocumentChunk(chunk_id="1", paper_id="1", title="T", text="evidence", page=2)
    assert grounding_gate([SearchHit(chunk=chunk, score=0.2)], 0.35) == (False, 0.2)
    assert grounding_gate([SearchHit(chunk=chunk, score=0.8)], 0.35) == (True, 0.8)


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
                executive_summary="Summary",
                key_findings=["Finding"],
                methods="Method",
                results=["Result"],
                limitations=["No explicit limitation identified."],
                follow_up_questions=["Question"],
                sources=[],
            )

    cache = Cache()
    gemini = Gemini()
    services = type("Services", (), {"briefing_cache": cache, "gemini": gemini})()
    state = {"classification": "briefing", "selected_papers": [paper], "query": "brief"}

    generate_response(state, services)
    result = generate_response(state, services)

    assert result["briefing"].title == "Cached"
    assert gemini.calls == 1
