"""Pure-ish LangGraph nodes for the research workflow."""

from __future__ import annotations

import re
from typing import Iterable

from app.services.config import Services
from app.services.arxiv import extract_arxiv_id
from app.state import CandidateScore, Paper, ResearchState, SearchHit, SourceMetadata

_STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "how", "in", "is", "of", "on",
    "or", "the", "this", "to", "what", "with", "paper", "papers", "about",
}


def _terms(value: str) -> list[str]:
    return [
        term for term in re.findall(r"[a-z0-9][a-z0-9-]*", value.lower())
        if term not in _STOP_WORDS and len(term) > 1
    ]


def score_paper(paper: Paper, query: str) -> CandidateScore:
    """Score a paper with separately inspectable title/abstract/category signals."""

    terms = list(dict.fromkeys(_terms(query)))
    title = set(_terms(paper.title))
    abstract = set(_terms(paper.summary))
    categories = set(_terms(" ".join(paper.categories)))
    matched = [term for term in terms if term in title or term in abstract or term in categories]
    title_score = sum(term in title for term in terms) / max(len(terms), 1)
    abstract_score = sum(term in abstract for term in terms) / max(len(terms), 1)
    category_score = sum(term in categories for term in terms) / max(len(terms), 1)
    # Titles are the strongest signal, while categories provide a useful tie-breaker.
    total = 0.55 * title_score + 0.30 * abstract_score + 0.15 * category_score
    if terms and " ".join(terms) in " ".join(_terms(paper.title)):
        total = min(1.0, total + 0.15)
    return CandidateScore(
        arxiv_id=paper.arxiv_id,
        title_relevance=round(title_score, 6),
        abstract_relevance=round(abstract_score, 6),
        category_relevance=round(category_score, 6),
        total=round(total, 6),
        matched_terms=matched,
    )


def rank_candidates(papers: Iterable[Paper], query: str) -> list[tuple[Paper, CandidateScore]]:
    ranked = [(paper, score_paper(paper, query)) for paper in papers]
    return sorted(
        ranked,
        key=lambda item: (-item[1].total, item[0].published or "", item[0].arxiv_id),
    )


def rerank_chunks(hits: Iterable[SearchHit], query: str, limit: int = 5) -> list[SearchHit]:
    """Apply a small lexical tie-breaker after vector retrieval."""

    terms = set(_terms(query))
    ranked: list[tuple[float, SearchHit]] = []
    for hit in hits:
        text_terms = set(_terms(hit.chunk.text))
        lexical = len(terms & text_terms) / max(len(terms), 1)
        combined = 0.8 * hit.score + 0.2 * lexical
        ranked.append((combined, hit.model_copy(update={"score": combined})))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]


def grounding_gate(
    hits: Iterable[SearchHit], threshold: float = 0.20
) -> tuple[bool, float]:
    """Decide whether retrieval is strong enough to call the language model."""

    scores: list[float] = []
    for hit in hits:
        try:
            scores.append(max(0.0, min(1.0, float(hit.score))))
        except (TypeError, ValueError):
            continue
    best = max(scores, default=0.0)
    return best >= threshold, best


def classify_query(state: ResearchState) -> dict:
    requested = state.get("classification")
    if requested in {"search", "briefing", "qa", "paper"}:
        return {"classification": requested, "errors": []}
    query = state.get("query", "").strip()
    lowered = query.lower()
    direct_id = extract_arxiv_id(query)
    if direct_id and (
        query.strip().lower() == direct_id.lower()
        or lowered.strip().startswith(("arxiv:", "arxiv "))
        or "arxiv.org" in query.lower()
        or re.fullmatch(r"\s*[\w.-]+/\d{7}(?:v\d+)?\s*", query, re.I)
    ):
        return {"classification": "paper", "direct_paper_id": direct_id, "errors": []}
    if re.match(r"^\s*(brief|briefing|overview|survey|compare)\b", lowered) or any(
        word in lowered for word in ("brief", "briefing", "overview", "survey", "compare")
    ):
        classification = "briefing"
    elif "?" in query or re.search(r"\b(what|why|how|who|when|where|which|explain)\b", lowered):
        classification = "qa"
    else:
        classification = "search"
    return {"classification": classification, "errors": []}


def search_papers(state: ResearchState, services: Services) -> dict:
    try:
        max_results = max(1, min(50, int(state.get("max_results", 5))))
        direct_id = (
            state.get("context_paper_id")
            or state.get("direct_paper_id")
            or extract_arxiv_id(state["query"])
        )
        if direct_id and state.get("classification") in {"paper", "briefing", "qa"}:
            resolver = (
                getattr(services.arxiv, "resolve", None)
                or getattr(services.arxiv, "get_paper", None)
                or getattr(services.arxiv, "get_by_id", None)
            )
            paper = resolver(direct_id) if resolver else None
            papers = [paper] if paper else []
            if not papers:
                return {
                    "papers": [],
                    "selected_papers": [],
                    "errors": [f"Could not resolve arXiv paper {direct_id}"],
                }
        else:
            papers = services.arxiv.search(state["query"], max_results=max_results)
        ranked = rank_candidates(papers, state["query"])
        scores = [score for _, score in ranked]
        ranked_papers = [paper for paper, _ in ranked]
        selected = ranked_papers[:1]
        return {"papers": ranked_papers, "selected_papers": selected, "candidate_scores": scores}
    except Exception as exc:
        return {"papers": [], "selected_papers": [], "errors": [f"arXiv search failed: {exc}"]}


def index_papers(state: ResearchState, services: Services) -> dict:
    if state.get("classification") == "search":
        return {"chunks": [], "errors": list(state.get("errors", []))}
    chunks = []
    errors = list(state.get("errors", []))
    for paper in state.get("selected_papers", []):
        try:
            services.pdf.download(paper)
            chunks.extend(services.pdf.chunks(paper))
        except Exception as exc:
            errors.append(f"Could not index {paper.arxiv_id}: {exc}")
    if chunks:
        try:
            services.vector.add(chunks)
        except Exception as exc:
            errors.append(f"Vector store failed: {exc}")
    return {"chunks": chunks, "errors": errors}


def retrieve_context(state: ResearchState, services: Services) -> dict:
    if state.get("classification") == "search":
        return {"retrieved": []}
    if not state.get("chunks"):
        return {"retrieved": []}
    try:
        selected = state.get("selected_papers", [])
        if not selected:
            return {"retrieved": []}
        retrieved = services.vector.query(
            state["query"], paper_id=selected[0].arxiv_id, n_results=10
        )
        reranked = rerank_chunks(retrieved, state["query"], limit=5)
        return {
            "retrieved_candidates": retrieved,
            "reranked_chunks": reranked,
            "retrieved": reranked,
            "collection_name": services.vector.collection_name(selected[0].arxiv_id),
        }
    except Exception as exc:
        return {"retrieved": [], "errors": [*state.get("errors", []), f"Retrieval failed: {exc}"]}


def generate_response(state: ResearchState, services: Services) -> dict:
    try:
        if state.get("classification") in {"briefing", "paper"}:
            if not state.get("selected_papers"):
                return {
                    "errors": [
                        *state.get("errors", []),
                        "No paper was selected for the briefing.",
                    ]
                }
            selected = state.get("selected_papers", [])
            cache = getattr(services, "briefing_cache", None)
            cache_key = f"{selected[0].arxiv_id}:{state['query']}" if selected else state["query"]
            cached = cache.get(cache_key) if cache else None
            if isinstance(cached, dict):
                from app.state import ExecutiveBriefing

                return {"briefing": ExecutiveBriefing.model_validate(cached)}
            briefing = services.gemini.briefing(selected, state.get("retrieved", []), state["query"])
            if cache:
                cache.set(cache_key, briefing.model_dump(mode="json"))
            return {"briefing": briefing}
        if state.get("classification") == "qa":
            try:
                threshold = float(
                    state.get(
                        "grounding_threshold",
                        getattr(services, "grounding_threshold", 0.20),
                    )
                )
            except (TypeError, ValueError):
                threshold = 0.20
            threshold = max(0.0, min(1.0, threshold))
            grounded, best = grounding_gate(state.get("retrieved", []), threshold)
            sources = [
                SourceMetadata(
                    arxiv_id=hit.chunk.paper_id,
                    chunk_id=hit.chunk.chunk_id,
                    title=hit.chunk.title,
                    page=hit.chunk.page,
                    section=hit.chunk.section,
                )
                for hit in state.get("retrieved", [])
            ]
            if not grounded:
                return {
                    "answer": (
                        "I couldn't find enough information in the paper to answer that."
                    ),
                    "grounded": False,
                    "grounding_score": best,
                    "sources": sources,
                }
            return {
                "answer": services.gemini.answer(state["query"], state.get("retrieved", [])),
                "grounded": True,
                "grounding_score": best,
                "sources": sources,
            }
        return {
            "answer": (
                "\n".join(
                    f"{index}. {paper.title} — {paper.abs_url}"
                    for index, paper in enumerate(state.get("papers", []), start=1)
                )
                or "No arXiv papers matched the request."
            )
        }
    except Exception as exc:
        return {"errors": [*state.get("errors", []), f"Generation failed: {exc}"]}
