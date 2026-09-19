import pytest

pytest.importorskip("langgraph")

from app.graph import build_graph
from app.services.config import Services
from app.state import DocumentChunk, Paper


class FakeArxiv:
    def search(self, query, max_results=5):
        return [Paper(arxiv_id="1", title="Demo", abs_url="https://arxiv.org/abs/1")]


class FakePdf:
    def __init__(self):
        self.downloaded = []

    def download(self, paper):
        self.downloaded.append(paper.arxiv_id)

    def chunks(self, paper):
        return [DocumentChunk(
            chunk_id=f"{paper.arxiv_id}:1:0",
            paper_id=paper.arxiv_id,
            title=paper.title,
            text="evidence",
            page=1,
        )]


class FakeVector:
    def add(self, chunks): pass
    def query(self, text, paper_id, n_results=6): return []


class FakeGemini:
    def answer(self, query, hits, history=None): return "grounded"
    def briefing(self, papers, hits, query): return "brief"


def test_graph_runs_without_network():
    result = build_graph(Services(FakeArxiv(), FakePdf(), FakeVector(), FakeGemini())).invoke(
        {"query": "demo"}
    )
    assert "Demo" in result["answer"]


def test_topic_selection_sends_only_top_ranked_paper_to_pdf():
    pdf = FakePdf()
    arxiv = type("RankedArxiv", (), {
        "search": lambda self, query, max_results=5: [
            Paper(
                arxiv_id="1",
                title="Graph neural networks",
                summary="graph neural networks",
                categories=["cs.LG"],
            ),
            Paper(
                arxiv_id="2",
                title="Unrelated systems",
                summary="systems",
                categories=["cs.OS"],
            ),
        ]
    })()
    result = build_graph(Services(arxiv, pdf, FakeVector(), FakeGemini())).invoke(
        {"query": "graph neural networks", "classification": "briefing"}
    )
    assert result["selected_papers"][0].arxiv_id == "1"
    assert pdf.downloaded == ["1"]
