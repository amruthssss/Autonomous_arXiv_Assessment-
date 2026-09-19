from pathlib import Path

import socket
import urllib.error

import pytest

from app.services.arxiv import ArxivAPIError, ArxivService


ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><id>http://arxiv.org/abs/1234.5678v1</id><title> Retrieval Systems </title>
<summary>retrieval paper</summary><published>2024-01-01T00:00:00Z</published>
<updated>2024-01-01T00:00:00Z</updated><author><name>Ada</name></author>
<link title="pdf" href="https://arxiv.org/pdf/1234.5678"/></entry></feed>"""


class Response:
    def __enter__(self): return self
    def __exit__(self, *args): return None
    def read(self): return ATOM.encode()


class EmptyResponse(Response):
    def read(self): return b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class StatusResponse(Response):
    def __init__(self, status: int):
        self.status = status


def http_error(status: int, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://export.arxiv.org/api/query",
        status,
        "failure",
        headers or {},
        None,
    )


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
    assert "sortBy=relevance" in request.full_url
    assert request.get_header("User-agent").startswith("arxiv-research-assistant/")
    assert request.get_header("Accept") == "application/atom+xml"
    assert timeout == 20.0


@pytest.mark.parametrize("max_results", [3, 10])
def test_arxiv_request_preserves_requested_max_results(tmp_path: Path, max_results: int):
    calls = []

    def opener(request, timeout=0):
        calls.append(request.full_url)
        return Response()

    ArxivService(tmp_path, opener=opener).search("transformer architecture", max_results)

    assert f"max_results={max_results}" in calls[0]


def test_request_pacing_is_sequential_and_configurable(tmp_path: Path, monkeypatch):
    calls = []
    sleeps = []
    clock = iter([10.0] * 12)
    monkeypatch.setattr("app.services.arxiv.time.monotonic", lambda: next(clock))

    def opener(request, timeout=0):
        calls.append(request.full_url)
        return EmptyResponse()

    service = ArxivService(
        tmp_path,
        opener=opener,
        min_request_interval=1.0,
        sleeper=sleeps.append,
    )
    service.search("first", 1)
    service.search("second", 1)
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_empty_success_response_is_a_valid_zero_result(tmp_path: Path):
    service = ArxivService(tmp_path, opener=lambda request, timeout=0: EmptyResponse())
    assert service.search("no matches", 3) == []


def test_non_200_response_object_uses_http_status_policy(tmp_path: Path):
    service = ArxivService(
        tmp_path,
        opener=lambda request, timeout=0: StatusResponse(503),
        max_retries=0,
    )
    with pytest.raises(ArxivAPIError, match="HTTP 503"):
        service.search("server failure", 1)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (400, "HTTP 400"),
        (403, "HTTP 403"),
        (404, "HTTP 404"),
        (406, "HTTP 406"),
    ],
)
def test_non_retryable_http_errors_are_explicit(
    tmp_path: Path, status: int, expected: str
):
    def opener(request, timeout=0):
        raise http_error(status)

    service = ArxivService(tmp_path, opener=opener, max_retries=2, sleeper=lambda _: None)
    with pytest.raises(ArxivAPIError, match=expected):
        service.search("failure", 1)


def test_406_uses_fallback_once_then_reports_failure(tmp_path: Path):
    calls = []

    def opener(request, timeout=0):
        calls.append(request.full_url)
        raise http_error(406)

    service = ArxivService(tmp_path, opener=opener, max_retries=0, sleeper=lambda _: None)
    with pytest.raises(ArxivAPIError, match="HTTP 406"):
        service.search("failure", 1)
    assert len(calls) == 2
    assert calls[1].startswith("https://arxiv.org/api/query?")


def test_406_fallback_can_recover_with_bounded_retry(tmp_path: Path):
    calls = []

    def opener(request, timeout=0):
        calls.append(request.full_url)
        if len(calls) < 3:
            raise http_error(406)
        return Response()

    service = ArxivService(tmp_path, opener=opener, sleeper=lambda _: None)
    assert service.search("fallback retry", 1)[0].arxiv_id == "1234.5678"
    assert len(calls) == 3
    assert all("api/query" in call for call in calls)


def test_failed_request_is_not_cached_as_empty_success(tmp_path: Path):
    attempts = []

    def failing(request, timeout=0):
        attempts.append("failure")
        raise http_error(406)

    service = ArxivService(tmp_path, opener=failing, max_retries=0, sleeper=lambda _: None)
    with pytest.raises(ArxivAPIError):
        service.search("uncached failure", 1)
    assert service.cache.get("uncached failure|1") is None

    def succeeding(request, timeout=0):
        attempts.append("success")
        return EmptyResponse()

    service.opener = succeeding
    assert service.search("uncached failure", 1) == []
    assert attempts[-1] == "success"


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_http_errors_retry_with_bound(tmp_path: Path, status: int):
    calls = []
    sleeps = []

    def opener(request, timeout=0):
        calls.append(request.full_url)
        raise http_error(status)

    service = ArxivService(
        tmp_path,
        opener=opener,
        max_retries=2,
        backoff_base=0.01,
        sleeper=sleeps.append,
    )
    with pytest.raises(ArxivAPIError, match=f"HTTP {status}"):
        service.search("failure", 1)
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_rate_limit_respects_retry_after(tmp_path: Path):
    sleeps = []
    attempts = 0

    def opener(request, timeout=0):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise http_error(429, {"Retry-After": "7"})
        return Response()

    service = ArxivService(tmp_path, opener=opener, sleeper=sleeps.append)
    assert service.search("retry", 1)[0].arxiv_id == "1234.5678"
    assert sleeps == [7.0]


@pytest.mark.parametrize("exc", [TimeoutError("slow"), socket.timeout("slow")])
def test_timeout_retries_then_reports_timeout(tmp_path: Path, exc: Exception):
    sleeps = []

    def opener(request, timeout=0):
        raise exc

    service = ArxivService(tmp_path, opener=opener, max_retries=1, sleeper=sleeps.append)
    with pytest.raises(ArxivAPIError, match="timed out"):
        service.search("timeout", 1)
    assert len(sleeps) == 1


def test_network_failure_retries_then_reports_network_error(tmp_path: Path):
    sleeps = []

    def opener(request, timeout=0):
        raise urllib.error.URLError("offline")

    service = ArxivService(tmp_path, opener=opener, max_retries=1, sleeper=sleeps.append)
    with pytest.raises(ArxivAPIError, match="Network failure"):
        service.search("offline", 1)
    assert len(sleeps) == 1
