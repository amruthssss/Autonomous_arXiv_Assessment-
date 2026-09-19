"""Official arXiv Atom API client with deterministic local ranking."""

from __future__ import annotations

import re
import random
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from app.services.cache import JsonCache
from app.state import Paper

ATOM = "{http://www.w3.org/2005/Atom}"
API_URL = "https://export.arxiv.org/api/query"
FALLBACK_API_URL = "https://arxiv.org/api/query"
USER_AGENT = "arxiv-research-assistant/1.0 (mailto:research-assistant@example.com)"
ARXIV_ID_RE = re.compile(
    r"(?<![\w.])(?P<id>(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?)(?!\w)",
    re.IGNORECASE,
)


class ArxivAPIError(RuntimeError):
    """A request-level failure from the arXiv API or its network path."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @classmethod
    def from_http(cls, status: int) -> "ArxivAPIError":
        messages = {
            400: "arXiv API rejected the request (HTTP 400).",
            403: "arXiv API access was forbidden (HTTP 403).",
            404: "The requested arXiv resource was not found (HTTP 404).",
            406: "arXiv API rejected the request (HTTP 406).",
            429: "arXiv API rate limit reached (HTTP 429). Please wait before retrying.",
        }
        message = messages.get(
            status,
            f"arXiv API/server error (HTTP {status}). "
            "The service may be temporarily unavailable. Please retry later.",
        )
        return cls(message, status=status)


def extract_arxiv_id(value: str) -> str | None:
    """Return a normalized arXiv identifier from an ID or arXiv URL."""

    text = value.strip()
    match = ARXIV_ID_RE.search(text)
    if not match:
        return None
    return re.sub(r"v\d+$", "", match.group("id"), flags=re.IGNORECASE)


def is_arxiv_reference(value: str) -> bool:
    return extract_arxiv_id(value) is not None and (
        bool(re.fullmatch(r"\s*[\w.-]+/\d{7}(?:v\d+)?\s*", value, re.I))
        or bool(re.fullmatch(r"\s*\d{4}\.\d{4,5}(?:v\d+)?\s*", value))
        or "arxiv.org" in value.lower()
    )


class ArxivService:
    """Search arXiv through export.arxiv.org and cache normalized results."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        opener: object | None = None,
        timeout: float = 20.0,
        max_retries: int = 2,
        backoff_base: float = 1.0,
        sleeper: Callable[[float], None] | None = None,
        min_request_interval: float = 0.0,
        debug: bool = False,
    ) -> None:
        self.cache = JsonCache(cache_dir, "arxiv")
        self.opener = opener
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.backoff_base = max(0.0, backoff_base)
        self.sleeper = sleeper or time.sleep
        self.min_request_interval = max(0.0, min_request_interval)
        self.debug = debug
        self._last_request_at = 0.0

    @staticmethod
    def _terms(query: str) -> list[str]:
        return re.findall(r"[a-z0-9][a-z0-9\-']*", query.lower())

    def _request(self, url: str) -> bytes:
        def request_for(target: str) -> urllib.request.Request:
            return urllib.request.Request(
                target,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/atom+xml",
                    "Accept-Encoding": "identity",
                },
            )

        def request_once(target: str) -> bytes:
            headers = {
                "User-Agent": USER_AGENT,
                "Accept": "application/atom+xml",
                "Accept-Encoding": "identity",
            }
            if self.opener is None:
                try:
                    import requests
                except ImportError as exc:
                    raise RuntimeError(
                        "requests is required for arXiv API access"
                    ) from exc
                try:
                    response = requests.get(
                        target,
                        headers=headers,
                        timeout=self.timeout,
                        allow_redirects=True,
                    )
                except requests.Timeout as exc:
                    raise TimeoutError(str(exc)) from exc
                except requests.RequestException as exc:
                    raise urllib.error.URLError(str(exc)) from exc
                if response.status_code != 200:
                    raise urllib.error.HTTPError(
                        target,
                        response.status_code,
                        f"HTTP {response.status_code}",
                        response.headers,
                        None,
                    )
                return response.content
            try:
                with self.opener(request_for(target), timeout=self.timeout) as response:
                    status = getattr(response, "status", 200)
                    if status != 200:
                        raise urllib.error.HTTPError(
                            target,
                            int(status),
                            f"HTTP {status}",
                            getattr(response, "headers", None),
                            None,
                        )
                    return response.read()
            except TypeError:
                # Makes simple test doubles which only accept a URL convenient.
                with self.opener(target) as response:  # type: ignore[operator]
                    return response.read()

        target = url
        fallback_used = False
        retries_used = 0
        while True:
            self._pace()
            started = time.monotonic()
            self._debug(
                f"[DEBUG] arXiv request\n"
                f"  url: {target}\n"
                f"  attempt: {retries_used + 1}/{self.max_retries + 1}"
            )
            try:
                payload = request_once(target)
                self._debug(
                    f"  status: 200\n"
                    f"  elapsed: {time.monotonic() - started:.3f}s"
                )
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code == 406 and target.startswith(API_URL) and not fallback_used:
                    target = FALLBACK_API_URL + target[len(API_URL):]
                    fallback_used = True
                    self._debug(
                        f"  status: 406\n"
                        f"  elapsed: {time.monotonic() - started:.3f}s\n"
                        "  retryable: fallback endpoint"
                    )
                    continue
                # A 406 from both official endpoints can be transient (the
                # same query may be accepted shortly afterward). Retry only
                # the fallback endpoint within the existing bounded policy.
                retryable = exc.code in {429, 500, 502, 503, 504} or (
                    exc.code == 406 and fallback_used
                )
                if not retryable:
                    self._debug(
                        f"  status: {exc.code}\n"
                        f"  elapsed: {time.monotonic() - started:.3f}s\n"
                        "  retryable: false"
                    )
                    raise ArxivAPIError.from_http(exc.code) from exc
                if retries_used >= self.max_retries:
                    self._debug(
                        f"  status: {exc.code}\n"
                        f"  elapsed: {time.monotonic() - started:.3f}s\n"
                        "  retryable: exhausted"
                    )
                    raise ArxivAPIError.from_http(exc.code) from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = max(0.0, float(retry_after)) if retry_after else 0.0
                except ValueError:
                    delay = 0.0
                if not delay:
                    delay = self.backoff_base * (2**retries_used)
                    delay += random.uniform(0.0, min(0.25, self.backoff_base))
                delay = min(delay, 30.0)
                self._debug(
                    f"  status: {exc.code}\n"
                    f"  elapsed: {time.monotonic() - started:.3f}s\n"
                    f"  retryable: true\n"
                    f"  sleep: {delay:.3f}s"
                )
                self.sleeper(delay)
                retries_used += 1
            except (TimeoutError, socket.timeout) as exc:
                if retries_used >= self.max_retries:
                    raise ArxivAPIError(
                        "Request to the arXiv API timed out. "
                        "Please check your network connection and try again."
                    ) from exc
                delay = min(
                    self.backoff_base * (2**retries_used)
                    + random.uniform(0.0, min(0.25, self.backoff_base)),
                    30.0,
                )
                self.sleeper(delay)
                retries_used += 1
            except urllib.error.URLError as exc:
                if retries_used >= self.max_retries:
                    raise ArxivAPIError(
                        f"Network failure while contacting the arXiv API: {exc.reason}"
                    ) from exc
                delay = min(
                    self.backoff_base * (2**retries_used)
                    + random.uniform(0.0, min(0.25, self.backoff_base)),
                    30.0,
                )
                self.sleeper(delay)
                retries_used += 1

    def _pace(self) -> None:
        if self.min_request_interval <= 0:
            self._last_request_at = time.monotonic()
            return
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_request_interval:
            self.sleeper(self.min_request_interval - elapsed)
        self._last_request_at = time.monotonic()

    def _debug(self, message: str) -> None:
        if self.debug:
            print(message, file=sys.stderr)

    @staticmethod
    def _parse(payload: bytes) -> list[Paper]:
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise RuntimeError("arXiv returned malformed XML") from exc
        papers: list[Paper] = []
        for entry in root.findall(f"{ATOM}entry"):
            identifier = extract_arxiv_id(entry.findtext(f"{ATOM}id") or "")
            if not identifier:
                continue
            title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
            summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
            links = entry.findall(f"{ATOM}link")
            pdf_url = next(
                (link.attrib.get("href", "") for link in links if link.attrib.get("title") == "pdf"),
                f"https://arxiv.org/pdf/{identifier}",
            )
            authors = [
                (author.findtext(f"{ATOM}name") or "").strip()
                for author in entry.findall(f"{ATOM}author")
            ]
            categories = [
                category.attrib.get("term", "").strip()
                for category in entry.findall(f"{ATOM}category")
                if category.attrib.get("term")
            ]
            papers.append(
                Paper(
                    arxiv_id=identifier,
                    title=title,
                    summary=summary,
                    authors=[author for author in authors if author],
                    published=entry.findtext(f"{ATOM}published") or "",
                    updated=entry.findtext(f"{ATOM}updated") or "",
                    pdf_url=pdf_url,
                    abs_url=f"https://arxiv.org/abs/{identifier}",
                    categories=categories,
                )
            )
        return papers

    def resolve(self, arxiv_id: str) -> Paper | None:
        """Resolve exactly one paper; never broaden a direct-paper request."""

        identifier = extract_arxiv_id(arxiv_id)
        if not identifier:
            raise ValueError(f"Not a valid arXiv identifier: {arxiv_id!r}")
        cache_key = f"id:{identifier}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            try:
                return Paper.model_validate(cached)
            except ValidationError:
                pass
        encoded = urllib.parse.quote(identifier)
        papers = self._parse(
            self._request(f"{API_URL}?id_list={encoded}")
        )
        paper = next((item for item in papers if item.arxiv_id == identifier), None)
        if paper:
            self.cache.set(cache_key, paper.model_dump())
        return paper

    # Backwards-compatible name for callers that prefer noun-based APIs.
    get_paper = resolve

    def search(self, query: str, max_results: int = 5) -> list[Paper]:
        normalized = " ".join(query.split())
        cache_key = f"relevance|{normalized}|{max_results}"
        cached = self.cache.get(cache_key)
        if isinstance(cached, list):
            try:
                return [Paper.model_validate(item) for item in cached]
            except ValidationError:
                # A partial/manual cache entry should never block a fresh search.
                pass

        if not normalized:
            raise ValueError("Search query cannot be empty")
        # arXiv's API query grammar requires each term to be scoped.  Joining
        # terms with AND avoids sending a free-form phrase as one field value.
        search_query = " AND ".join(f"all:{term}" for term in self._terms(normalized))
        query_params = urllib.parse.urlencode(
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            quote_via=urllib.parse.quote,
        )
        url = f"{API_URL}?{query_params}"
        papers = self._parse(self._request(url))
        terms = self._terms(normalized)
        self.cache.set(cache_key, [paper.model_dump() for paper in papers])
        return papers
