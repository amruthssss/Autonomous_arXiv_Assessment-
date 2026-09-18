"""Official arXiv Atom API client with deterministic local ranking."""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
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
    ) -> None:
        self.cache = JsonCache(cache_dir, "arxiv")
        self.opener = opener or urllib.request.urlopen
        self.timeout = timeout

    @staticmethod
    def _terms(query: str) -> list[str]:
        return re.findall(r"[a-z0-9][a-z0-9\-']*", query.lower())

    def _request(self, url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/atom+xml",
                "Accept-Encoding": "identity",
            },
        )
        for attempt in range(3):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code != 406:
                    raise
                if attempt == 2 and url.startswith(API_URL):
                    fallback_url = FALLBACK_API_URL + url[len(API_URL):]
                    with self.opener(
                        urllib.request.Request(
                            fallback_url,
                            headers={
                                "User-Agent": USER_AGENT,
                                "Accept": "application/atom+xml",
                                "Accept-Encoding": "identity",
                            },
                        ),
                        timeout=self.timeout,
                    ) as response:
                        return response.read()
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
            except TypeError:
                # Makes simple test doubles which only accept a URL convenient.
                with self.opener(url) as response:  # type: ignore[operator]
                    return response.read()
        raise RuntimeError("arXiv request failed after retries")

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
        cache_key = f"{normalized}|{max_results}"
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
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            quote_via=urllib.parse.quote,
        )
        url = f"{API_URL}?{query_params}"
        papers = self._parse(self._request(url))
        terms = self._terms(normalized)
        papers.sort(
            key=lambda paper: (
                sum(paper.title.lower().count(term) * 3 for term in terms)
                + sum(paper.summary.lower().count(term) for term in terms),
                paper.published,
                paper.arxiv_id,
            ),
            reverse=True,
        )
        self.cache.set(cache_key, [paper.model_dump() for paper in papers])
        return papers
