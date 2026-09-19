"""Typed state and domain models shared by the graph nodes."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """An arXiv record and the metadata needed for grounding."""

    arxiv_id: str
    title: str
    summary: str = ""
    authors: list[str] = Field(default_factory=list)
    published: str = ""
    updated: str = ""
    pdf_url: str = ""
    abs_url: str = ""
    categories: list[str] = Field(default_factory=list)


class SourceMetadata(BaseModel):
    """A citation pointing at an attributable location in a paper."""

    arxiv_id: str
    chunk_id: str = ""
    title: str = ""
    page: int = 0
    section: str = ""
    abs_url: str = ""


class CandidateScore(BaseModel):
    """The explainable score used to choose a topic's representative paper."""

    arxiv_id: str
    title_relevance: float
    abstract_relevance: float
    category_relevance: float
    total: float
    matched_terms: list[str] = Field(default_factory=list)

    @property
    def title_score(self) -> float:
        return self.title_relevance

    @property
    def abstract_score(self) -> float:
        return self.abstract_relevance

    @property
    def category_score(self) -> float:
        return self.category_relevance


class ExecutiveBriefing(BaseModel):
    """Structured, source-grounded output for a paper briefing."""

    title: str
    authors: list[str] = Field(default_factory=list)
    arxiv_id: str
    published: str = ""
    link: str = ""
    executive_summary: str
    problem_statement: str
    key_findings: list[str]
    methods: str
    results: list[str]
    limitations: list[str]
    follow_up_questions: list[str]
    sources: list[SourceMetadata]


class DocumentChunk(BaseModel):
    """A small, attributable section of a downloaded paper."""

    chunk_id: str
    paper_id: str
    title: str
    text: str
    page: int = 0
    start: int = 0
    end: int = 0
    section: str = ""


class SearchHit(BaseModel):
    chunk: DocumentChunk
    score: float


def _replace(_: Any, new: Any) -> Any:
    """LangGraph reducer which makes each node's value authoritative."""

    return new


class ResearchState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    query: str
    classification: Literal["search", "briefing", "qa", "paper"]
    search_status: Literal[
        "not_run", "success", "no_results", "arxiv_unavailable", "error"
    ]
    paper_status: Literal["not_found", "found", "ready"]
    fetch_status: Literal["not_run", "success", "failed"]
    parse_status: Literal["not_run", "success", "failed"]
    index_status: Literal["not_run", "ready", "failed"]
    max_results: int
    papers: Annotated[list[Paper], _replace]
    selected_papers: Annotated[list[Paper], _replace]
    candidate_scores: Annotated[list[CandidateScore], _replace]
    direct_paper_id: str
    context_paper_id: str
    chunks: Annotated[list[DocumentChunk], _replace]
    retrieved: Annotated[list[SearchHit], _replace]
    answer: str | None
    briefing: ExecutiveBriefing | str
    errors: Annotated[list[str], _replace]
    from_cache: bool
    grounding_threshold: float
    grounding_score: float
    grounded: bool
    conversation_history: list[dict[str, str]]
    active_paper_id: str
    collection_name: str
    paper_index_status: Literal[
        "not_checked", "indexed", "not_indexed", "corrupted", "not_found"
    ]
    index_diagnostic: str
    sources: Annotated[list[SourceMetadata], _replace]
    debug: bool
