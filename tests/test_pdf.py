from pathlib import Path

from app.services.pdf import PdfService
from app.state import Paper


def test_pdf_chunking_uses_stable_ids(tmp_path: Path, monkeypatch):
    service = PdfService(tmp_path)
    paper = Paper(arxiv_id="x/1", title="A")
    monkeypatch.setattr(service, "extract", lambda path: [(2, "one " * 100)])
    chunks = service.chunks(paper, chunk_size=30, overlap=5)
    assert chunks
    assert chunks[0].chunk_id == "x/1:2:0"
    assert chunks[0].page == 2


def test_pdf_validation_rejects_non_pdf(tmp_path: Path):
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"not a pdf")
    assert not PdfService.validate_pdf(path)
