from app.main import _print_result
from app.state import CandidateScore, Paper, SourceMetadata


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
    assert "Graph abstract" in output
    assert "Authors:" in output
    assert "Final Score : 0.870" in output
    assert "RANKING RESULTS" in output
    assert "SELECTED PAPER" in output
    assert "arXiv ID: 1" in output


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
    assert "Page 3 — Architecture — 1706.03762:3:7 — arXiv:1706.03762" in capsys.readouterr().out
