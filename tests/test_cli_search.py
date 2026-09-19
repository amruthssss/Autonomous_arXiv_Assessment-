from app.main import _briefing_markdown, _print_result
from app.state import CandidateScore, ExecutiveBriefing, Paper, SourceMetadata


def test_topic_search_output_includes_metadata_scores_and_selection(capsys):
    papers = [
        Paper(
            arxiv_id="1",
            title="Graph paper",
            authors=["Ada"],
            summary="Graph abstract",
            published="2024-01-01",
            categories=["cs.LG"],
            pdf_url="https://arxiv.org/pdf/1",
            abs_url="https://arxiv.org/abs/1",
        ),
        Paper(arxiv_id="2", title="Other paper"),
    ]
    scores = [
        CandidateScore(
            arxiv_id="1",
            title_relevance=1.0,
            abstract_relevance=0.8,
            category_relevance=0.5,
            total=0.87,
        ),
        CandidateScore(
            arxiv_id="2",
            title_relevance=0.0,
            abstract_relevance=0.0,
            category_relevance=0.0,
            total=0.0,
        ),
    ]
    _print_result(
        {
            "classification": "search",
            "query": "graph",
            "papers": papers,
            "candidate_scores": scores,
        }
    )
    output = capsys.readouterr().out
    assert "Graph paper" in output
    assert "Relevance: 0.870" in output
    assert "Graph abstract" not in output
    assert "RANKING RESULTS" not in output
    assert "SELECTED PAPER" in output
    assert "arXiv ID: 1" in output
    assert "Best match automatically selected: [1]" in output
    assert "Next command:" in output
    assert "select <number>" in output
    assert "Next:" not in output


def test_answer_output_includes_chunk_source_metadata(capsys):
    _print_result(
        {
            "answer": "Grounded answer.",
            "sources": [
                SourceMetadata(
                    arxiv_id="1706.03762",
                    chunk_id="1706.03762:3:7",
                    page=3,
                    section="Architecture",
                )
            ],
        }
    )
    assert "Page 3 — Architecture" in capsys.readouterr().out


def test_grounded_answer_output_includes_status(capsys):
    _print_result({"answer": "Grounded answer.", "grounded": True, "sources": []})
    assert "QA_STATUS: GROUNDED" in capsys.readouterr().out


def test_failed_search_is_not_rendered_as_zero_candidates(capsys):
    _print_result(
        {
            "classification": "search",
            "query": "graph neural networks",
            "papers": [],
            "search_status": "error",
            "errors": ["arXiv search failed: arXiv API rejected the request (HTTP 406)."],
        }
    )
    output = capsys.readouterr().out
    assert "SEARCH_STATUS: ERROR" in output
    assert "HTTP 406" in output
    assert "Found 0 candidate(s)." not in output


def test_failed_search_json_has_no_success_empty_answer(capsys):
    _print_result(
        {
            "classification": "search",
            "search_status": "error",
            "papers": [],
            "answer": None,
            "errors": ["arXiv search failed: HTTP 406"],
        },
        as_json=True,
    )
    output = capsys.readouterr().out
    assert '"search_status": "error"' in output
    assert '"answer": null' in output
    assert "No arXiv papers matched the request." not in output


def test_successful_zero_result_search_is_still_rendered_as_zero_candidates(capsys):
    _print_result(
        {
            "classification": "search",
            "query": "no matches",
            "papers": [],
            "search_status": "success",
            "candidate_scores": [],
        }
    )
    output = capsys.readouterr().out
    assert "Found 0 paper(s)." in output
    assert "ARXIV SEARCH FAILED" not in output


def test_search_debug_renders_diagnostics(capsys):
    _print_result(
        {
            "classification": "search",
            "query": "transformer architecture",
            "papers": [],
            "candidate_scores": [],
            "search_status": "error",
            "errors": ["arXiv search failed: HTTP 406"],
            "debug": True,
        }
    )
    output = capsys.readouterr().out
    assert "[DEBUG] Retrieval diagnostics" in output


def test_debug_output_includes_retrieval_scores_and_grounding(capsys):
    from app.state import DocumentChunk, SearchHit

    _print_result(
        {
            "answer": "Grounded answer.",
            "grounded": True,
            "grounding_score": 0.82,
            "retrieved": [
                SearchHit(
                    chunk=DocumentChunk(
                        chunk_id="chunk-1",
                        paper_id="1706.03762",
                        title="Paper",
                        text="Evidence",
                        page=3,
                        section="Method",
                    ),
                    score=0.82,
                )
            ],
            "debug": True,
        }
    )
    output = capsys.readouterr().out
    assert "[DEBUG] Retrieval diagnostics" in output
    assert "Retrieved chunks: 1" in output
    assert "score=0.8200" in output
    assert "Grounding score: 0.8200" in output


def test_briefing_markdown_contains_sections_and_sources():
    markdown = _briefing_markdown(
        ExecutiveBriefing(
            title="Test Paper",
            authors=["Ada"],
            arxiv_id="1706.03762",
            executive_summary="Summary",
            problem_statement="Problem",
            methods="Method",
            key_findings=["Finding"],
            results=["Result"],
            limitations=["Limitation"],
            follow_up_questions=["Question"],
            sources=[
                SourceMetadata(
                    arxiv_id="1706.03762",
                    page=3,
                    section="Method",
                )
            ],
        )
    )
    assert "# Test Paper" in markdown
    assert "## Executive summary" in markdown
    assert "- Finding" in markdown
    assert "Page 3 — Method — arXiv:1706.03762" in markdown
