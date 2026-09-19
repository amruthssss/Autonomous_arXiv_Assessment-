"""Pure-ish LangGraph nodes for the research workflow."""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import ValidationError

from app.services.config import Services
from app.services.arxiv import ArxivAPIError, extract_arxiv_id
from app.state import CandidateScore, Paper, ResearchState, SearchHit, SourceMetadata

_STOP_WORDS = {
    "a", "an", "and", "are", "for", "from", "how", "in", "is", "of", "on",
    "or", "the", "this", "to", "what", "with", "paper", "papers", "about",
}

_REFUSAL_MARKERS = (
    "couldn't find enough information",
    "could not find enough information",
    "evidence is insufficient",
    "insufficient evidence",
    "cannot answer from the provided excerpts",
    "can't answer from the provided excerpts",
)


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
                    "search_status": "no_results",
                    "paper_status": "not_found",
                    "paper_index_status": "not_found",
                    "errors": [f"Could not resolve arXiv paper {direct_id}"],
                }
        else:
            papers = services.arxiv.search(state["query"], max_results=max_results)
        ranked = rank_candidates(papers, state["query"])
        scores = [score for _, score in ranked]
        ranked_papers = [paper for paper, _ in ranked]
        selected = ranked_papers[:1]
        return {
            "papers": ranked_papers,
            "selected_papers": selected,
            "candidate_scores": scores,
            "search_status": "success" if ranked_papers else "no_results",
            "paper_status": "found" if selected else "not_found",
        }
    except ArxivAPIError as exc:
        return {
            "papers": [],
            "selected_papers": [],
            "search_status": "arxiv_unavailable",
            "paper_status": "not_found",
            "errors": [str(exc)],
        }
    except Exception as exc:
        return {
            "papers": [],
            "selected_papers": [],
            "search_status": "error",
            "paper_status": "not_found",
            "errors": [f"arXiv search failed: {exc}"],
        }


def index_papers(state: ResearchState, services: Services) -> dict:
    if state.get("classification") == "search" or state.get("search_status") in {
        "arxiv_unavailable",
        "error",
        "no_results",
    }:
        return {
            "chunks": [],
            "paper_index_status": state.get("paper_index_status", "not_checked"),
            "errors": list(state.get("errors", [])),
        }
    chunks = []
    errors = list(state.get("errors", []))
    fetch_status = "not_run"
    parse_status = "not_run"
    for paper in state.get("selected_papers", []):
        try:
            inspect = getattr(services.vector, "inspect_index", None)
            if inspect:
                status, diagnostic = inspect(paper.arxiv_id)
                if status == "indexed":
                    return {
                        "chunks": [],
                        "paper_status": "ready",
                        "fetch_status": "success",
                        "parse_status": "success",
                        "index_status": "ready",
                        "paper_index_status": "indexed",
                        "index_diagnostic": diagnostic,
                        "collection_name": services.vector.collection_name(paper.arxiv_id),
                        "errors": errors,
                    }
                if status == "corrupted":
                    errors.append(diagnostic)
                    delete_collection = getattr(services.vector, "delete_collection", None)
                    if delete_collection:
                        delete_collection(paper.arxiv_id)
            services.pdf.download(paper)
            fetch_status = "success"
            chunks.extend(services.pdf.chunks(paper))
            parse_status = "success"
        except Exception as exc:
            errors.append(f"Could not index {paper.arxiv_id}: {exc}")
            if fetch_status != "success":
                fetch_status = "failed"
            if parse_status != "success":
                parse_status = "failed"
    if chunks:
        try:
            services.vector.add(chunks)
        except Exception as exc:
            errors.append(f"Vector store failed: {exc}")
            return {
                "chunks": chunks,
                "paper_status": "found",
                "fetch_status": fetch_status,
                "parse_status": parse_status,
                "index_status": "failed",
                "paper_index_status": "not_indexed",
                "index_diagnostic": (
                    "The paper was downloaded and parsed successfully, "
                    "but local indexing could not be completed. "
                    "Run fetch again to retry."
                ),
                "errors": errors,
            }
    status = "indexed" if chunks and not errors else "not_indexed"
    return {
        "chunks": chunks,
        "paper_status": "found",
        "fetch_status": fetch_status,
        "parse_status": parse_status,
        "index_status": "ready" if status == "indexed" else "failed",
        "paper_index_status": status,
        "index_diagnostic": (
            "Paper is not indexed. Fetching/parsing PDF and building a local index."
            if status == "not_indexed"
            else ""
        ),
        "errors": errors,
    }


def retrieve_context(state: ResearchState, services: Services) -> dict:
    if state.get("classification") == "search":
        return {"retrieved": []}
    if state.get("search_status") == "error":
        return {"retrieved": []}
    if not state.get("chunks") and state.get("paper_index_status") != "indexed":
        return {"retrieved": []}
    try:
        selected = state.get("selected_papers", [])
        if not selected:
            return {"retrieved": []}
        retrieval_query = state["query"]
        if state.get("classification") in {"briefing", "paper"}:
            paper = selected[0]
            retrieval_query = f"{paper.title}. {paper.summary}"
        retrieved = services.vector.query(
            retrieval_query, paper_id=selected[0].arxiv_id, n_results=5
        )
        return {
            "retrieved": retrieved,
            "collection_name": services.vector.collection_name(selected[0].arxiv_id),
        }
    except Exception as exc:
        return {"retrieved": [], "errors": [*state.get("errors", []), f"Retrieval failed: {exc}"]}


def generate_response(state: ResearchState, services: Services) -> dict:
    try:
        if state.get("search_status") in {
            "no_results",
            "arxiv_unavailable",
            "error",
        }:
            return {"answer": None, "errors": list(state.get("errors", []))}
        if state.get("classification") in {"briefing", "paper"}:
            if not state.get("selected_papers"):
                return {
                    "errors": [
                        *state.get("errors", []),
                        "No paper was selected for the briefing.",
                    ]
                }
            selected = state.get("selected_papers", [])
            threshold = getattr(services, "grounding_threshold", 0.20)
            try:
                threshold = max(0.0, min(1.0, float(threshold)))
            except (TypeError, ValueError):
                threshold = 0.20
            grounded, best = grounding_gate(state.get("retrieved", []), threshold)
            if not grounded:
                return {
                    "briefing": (
                        "I couldn't find enough information in the paper to generate "
                        "a grounded briefing."
                    ),
                    "grounded": False,
                    "grounding_score": best,
                    "sources": [],
                }
            cache = getattr(services, "briefing_cache", None)
            cache_key = f"{selected[0].arxiv_id}:{state['query']}" if selected else state["query"]
            cached = cache.get(cache_key) if cache else None
            if isinstance(cached, dict):
                from app.state import ExecutiveBriefing

                try:
                    return {"briefing": ExecutiveBriefing.model_validate(cached)}
                except ValidationError:
                    cached = None
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
            answer = services.gemini.answer(
                state["query"],
                state.get("retrieved", []),
                state.get("conversation_history", []),
            )
            if any(marker in answer.lower() for marker in _REFUSAL_MARKERS):
                return {
                    "answer": "I couldn't find enough information in this paper to answer that.",
                    "grounded": False,
                    "grounding_score": best,
                    "sources": sources,
                }
            return {
                "answer": answer,
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
