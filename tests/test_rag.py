from app.nodes import generate_response, index_papers, retrieve_context
from app.services.gemini import GeminiService
from app.services.vector_store import ChromaVectorStore
from app.state import DocumentChunk, ExecutiveBriefing, Paper, SearchHit


def test_gemini_prompt_contains_retrieved_chunks_not_full_pdf():
    captured = {}

    class Models:
        def generate_content(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"text": "grounded answer"})()

    service = GeminiService(api_key="test-key")
    service._client = type("Client", (), {"models": Models()})()
    retrieved = SearchHit(
        chunk=DocumentChunk(
            chunk_id="1",
            paper_id="2005.11401",
            title="RAG",
            text="RETRIEVED CHUNK TEXT",
            page=1,
        ),
        score=0.8,
    )
    full_pdf_text = "FULL PDF TEXT MUST NOT BE INCLUDED"

    answer = service.answer("What is the method?", [retrieved])

    assert answer == "grounded answer"
    assert "RETRIEVED CHUNK TEXT" in captured["contents"]
    assert full_pdf_text not in captured["contents"]


def test_chroma_query_is_filtered_to_requested_paper(tmp_path):
    requested = "2005.11401"
    captured = {}

    class Embeddings:
        def embed(self, text):
            return [1.0, 0.0]

    class Collection:
        def query(self, **kwargs):
            captured.update(kwargs)
            return {
                "ids": [[f"{requested}:1:0"]],
                "documents": [["RAG evidence"]],
                "metadatas": [[{"paper_id": requested, "title": "RAG", "page": 1}]],
                "distances": [[0.1]],
            }

    store = ChromaVectorStore(tmp_path, Embeddings())
    store._collections[store.collection_name(requested)] = Collection()
    hits = store.query("What is the method?", requested, n_results=5)

    assert captured["where"] == {"paper_id": requested}
    assert hits
    assert all(hit.chunk.paper_id == requested for hit in hits)


def test_briefing_reuses_valid_index_when_pdf_is_unavailable():
    paper = Paper(arxiv_id="2005.11401", title="RAG", summary="Retrieval augmented generation")
    evidence = DocumentChunk(
        chunk_id="2005.11401:1:0",
        paper_id=paper.arxiv_id,
        title=paper.title,
        text="The paper proposes retrieval augmented generation.",
        page=1,
    )

    class Pdf:
        def download(self, _paper):
            raise AssertionError("A valid index should avoid PDF download")

        def chunks(self, _paper):
            raise AssertionError("A valid index should avoid PDF parsing")

    class Vector:
        def inspect_index(self, paper_id):
            return "indexed", "Paper already indexed locally."

        def collection_name(self, paper_id):
            return "paper_2005_11401"

        def query(self, text, paper_id, n_results=5):
            return [SearchHit(chunk=evidence, score=0.9)]

    class Gemini:
        def briefing(self, papers, hits, query):
            return ExecutiveBriefing(
                title=paper.title,
                arxiv_id=paper.arxiv_id,
                executive_summary="Summary",
                problem_statement="Problem",
                key_findings=["Finding"],
                methods="Method",
                results=["Result"],
                limitations=["Limitation"],
                follow_up_questions=["Question"],
                sources=[],
            )

    services = type(
        "Services",
        (),
        {
            "pdf": Pdf(),
            "vector": Vector(),
            "gemini": Gemini(),
            "grounding_threshold": 0.2,
        },
    )()
    indexed = index_papers(
        {"classification": "briefing", "search_status": "success", "selected_papers": [paper]},
        services,
    )
    retrieved = retrieve_context(
        {
            "classification": "briefing",
            "query": "2005.11401",
            "selected_papers": [paper],
            **indexed,
        },
        services,
    )
    generated = generate_response(
        {
            "classification": "briefing",
            "query": "2005.11401",
            "selected_papers": [paper],
            **indexed,
            **retrieved,
        },
        services,
    )

    assert indexed["paper_status"] == "ready"
    assert indexed["index_status"] == "ready"
    assert generated["briefing"].arxiv_id == paper.arxiv_id


def test_repeated_questions_retrieve_again_for_the_active_paper():
    calls = []
    paper = Paper(arxiv_id="1706.03762", title="Attention Is All You Need")

    class Vector:
        def query(self, text, paper_id, n_results=5):
            calls.append((text, paper_id, n_results))
            return []

        def collection_name(self, paper_id):
            return f"paper_{paper_id.replace('.', '_')}"

    services = type("Services", (), {"vector": Vector()})()
    state = {
        "classification": "qa",
        "selected_papers": [paper],
        "paper_index_status": "indexed",
        "query": "What is the contribution?",
    }
    retrieve_context(state, services)
    state["query"] = "What are the limitations?"
    retrieve_context(state, services)

    assert calls == [
        ("What is the contribution?", "1706.03762", 5),
        ("What are the limitations?", "1706.03762", 5),
    ]


def test_switching_active_paper_changes_retrieval_collection():
    calls = []

    class Vector:
        def query(self, text, paper_id, n_results=5):
            calls.append(paper_id)
            return []

        def collection_name(self, paper_id):
            return f"paper_{paper_id.replace('.', '_')}"

    services = type("Services", (), {"vector": Vector()})()
    for paper_id in ("1706.03762", "2005.11401"):
        retrieve_context(
            {
                "classification": "qa",
                "query": "What is the main contribution?",
                "selected_papers": [Paper(arxiv_id=paper_id, title="Paper")],
                "paper_index_status": "indexed",
            },
            services,
        )

    assert calls == ["1706.03762", "2005.11401"]
