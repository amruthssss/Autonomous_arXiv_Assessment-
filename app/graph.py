"""LangGraph assembly for the end-to-end research workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.nodes import (
    classify_query,
    generate_response,
    index_papers,
    retrieve_context,
    search_papers,
)
from app.services.config import Services, build_services
from app.state import ResearchState


def build_graph(services: Services | None = None):
    dependencies = services or build_services()
    graph = StateGraph(ResearchState)
    graph.add_node("classify", classify_query)
    graph.add_node("search", lambda state: search_papers(state, dependencies))
    graph.add_node("index", lambda state: index_papers(state, dependencies))
    graph.add_node("retrieve", lambda state: retrieve_context(state, dependencies))
    graph.add_node("generate", lambda state: generate_response(state, dependencies))
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "search")
    graph.add_edge("search", "index")
    graph.add_edge("index", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()

