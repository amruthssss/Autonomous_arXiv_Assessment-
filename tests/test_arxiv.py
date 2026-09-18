from pathlib import Path

from app.services.arxiv import ArxivService


ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/1234.5678v1</id><title> Retrieval Systems </title>
<summary>retrieval paper</summary><published>2024-01-01T00:00:00Z</published>
<updated>2024-01-01T00:00:00Z</updated><author><name>Ada</name></author>
<link title="pdf" href="https://arxiv.org/pdf/1234.5678"/></entry></feed>"""


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def read(self): return ATOM.encode()


def test_arxiv_parses_and_caches(tmp_path: Path):
    calls = []

    def opener(request, timeout=0):
        calls.append(request)
        return Response()

    service = ArxivService(tmp_path, opener=opener)
    papers = service.search("retrieval", 2)
    assert papers[0].arxiv_id == "1234.5678"
    assert papers[0].authors == ["Ada"]
    assert len(calls) == 1
    assert service.search("retrieval", 2)[0].title == "Retrieval Systems"
    assert len(calls) == 1


def test_arxiv_request_uses_official_endpoint_headers_and_query(tmp_path: Path):
    calls = []

    def opener(request, timeout=0):
        calls.append((request, timeout))
        return Response()

    service = ArxivService(tmp_path, opener=opener)
    service.search("graph neural networks", 5)

    request, timeout = calls[0]
    assert request.full_url.startswith("https://export.arxiv.org/api/query?")
    assert "search_query=all%3Agraph%20AND%20all%3Aneural%20AND%20all%3Anetworks" in request.full_url
    assert request.get_header("User-agent").startswith("arxiv-research-assistant/")
    assert request.get_header("Accept") == "application/atom+xml"
    assert timeout == 20.0
