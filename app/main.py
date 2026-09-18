"""Command-line interface for the grounded arXiv research assistant."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.graph import build_graph
from app.services.arxiv import extract_arxiv_id, is_arxiv_reference
from app.services.config import build_services
from app.services.session import ActivePaperStore
from app.state import ExecutiveBriefing


def _session_store() -> ActivePaperStore:
    return ActivePaperStore(Path(__file__).resolve().parents[1] / "data" / "session.json")


def _save_active(result: dict[str, Any]) -> None:
    selected = result.get("selected_papers", [])
    if not selected:
        return
    paper = selected[0]
    services = build_services()
    _session_store().save(
        paper.arxiv_id,
        services.vector.collection_name(paper.arxiv_id),
        paper.title,
        paper.abs_url,
    )


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description="Grounded arXiv research assistant")
    sub = cli.add_subparsers(dest="command")
    prompt = sub.add_parser("prompt", help="Interactive briefing and Q&A session")
    prompt.add_argument("query", nargs="?", help="Optional initial topic or paper")
    remove = sub.add_parser("remove", help="Remove local embeddings for a paper")
    remove.add_argument("query", help="arXiv ID or URL")
    for command, help_text in (
        ("search", "Search arXiv"),
        ("brief", "Create a cited paper briefing"),
        ("ask", "Ask a cited question about papers"),
        ("qa", "Ask a cited question about papers (alias for ask)"),
    ):
        command_parser = sub.add_parser(command, help=help_text)
        command_parser.add_argument("query")
        command_parser.add_argument("-n", "--max-results", type=int, default=5)
        command_parser.add_argument("--paper", help="arXiv ID or URL to use as context")
        command_parser.add_argument("--json", action="store_true", dest="as_json")
        command_parser.add_argument("--debug", action="store_true", help="Show retrieval diagnostics")
    return cli


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def _display(value: str, *, fallback: str = "Not available", limit: int | None = None) -> str:
    text = " ".join(value.split()) if value else fallback
    if limit and len(text) > limit:
        return f"{text[:limit].rstrip()}..."
    return text


def _print_topic_search(result: dict[str, Any]) -> None:
    papers = result.get("papers", [])
    scores = result.get("candidate_scores", [])
    query = result.get("query", "")
    print("=" * 60)
    print("ARXIV TOPIC SEARCH")
    print("=" * 60)
    print(f"\nQuery: {query}")
    print(f"\nFound {len(papers)} candidate paper(s).")
    for index, paper in enumerate(papers, start=1):
        score = scores[index - 1] if index - 1 < len(scores) else None
        print("\n" + "-" * 60)
        print(f"CANDIDATE {index}")
        print("-" * 60)
        print(f"\nTitle:\n{_display(paper.title)}")
        print(f"\nAuthors:\n{_display(', '.join(paper.authors))}")
        print(f"\nAbstract:\n{_display(paper.summary, limit=700)}")
        print(f"\narXiv ID:\n{_display(paper.arxiv_id)}")
        print(f"\nPublished:\n{_display(paper.published)}")
        print(f"\nCategories:\n{_display(', '.join(paper.categories))}")
        print(f"\nPDF:\n{_display(paper.pdf_url)}")
        print(f"\nAbstract URL:\n{_display(paper.abs_url)}")
        if score is not None:
            print("\nRELEVANCE SCORE")
            print(f"  Title       : {score.title_relevance:.3f}")
            print(f"  Abstract    : {score.abstract_relevance:.3f}")
            print(f"  Category    : {score.category_relevance:.3f}")
            print("  -----------------------")
            print(f"  Final Score : {score.total:.3f}")
    print("\n" + "=" * 60)
    print("RANKING RESULTS")
    print("=" * 60)
    for index, paper in enumerate(papers, start=1):
        score = scores[index - 1] if index - 1 < len(scores) else None
        final = f"{score.total:.3f}" if score is not None else "Not available"
        print(f"\nRank {index}")
        print(f"  Title: {_display(paper.title)}")
        print(f"  arXiv ID: {_display(paper.arxiv_id)}")
        print(f"  Score: {final}")
    if papers:
        selected = papers[0]
        score = scores[0] if scores else None
        final = f"{score.total:.3f}" if score is not None else "Not available"
        print("\n" + "=" * 60)
        print("SELECTED PAPER")
        print("=" * 60)
        print(f"\nSelected Paper:\n{_display(selected.title)}")
        print(f"\narXiv ID: {selected.arxiv_id}")
        print(f"Score: {final}")
        print(f'\nSelection: Highest deterministic relevance score for "{query}".')
        print("\nNext:\n→ Fetch PDF\n→ Parse PDF\n→ Chunk\n→ Generate embeddings\n→ Store in ChromaDB")


def _print_result(result: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, default=_jsonable, indent=2))
        return
    if result.get("classification") == "search":
        _print_topic_search(result)
        for error in result.get("errors", []):
            print(f"warning: {error}")
        return
    briefing = result.get("briefing")
    if isinstance(briefing, ExecutiveBriefing):
        selected = result.get("selected_papers", [])
        paper = selected[0] if selected else None
        print(f"# {briefing.title}")
        if paper:
            print(f"\nAuthors: {', '.join(paper.authors)}")
            print(f"arXiv ID: {paper.arxiv_id}")
            print(f"Published: {paper.published}")
            print(f"Link: {paper.abs_url}")
        print(f"\nWhy it matters:\n{briefing.executive_summary}")
        print(f"\nMethod:\n{briefing.methods}")
        print("\nKey findings:")
        for finding in briefing.key_findings:
            print(f"- {finding}")
        print("\nLimitations:")
        for limitation in briefing.limitations:
            print(f"- {limitation}")
        print("\nFollow-up questions:")
        for question in briefing.follow_up_questions:
            print(f"- {question}")
    else:
        print(briefing or result.get("answer") or result.get("search_results", ""))
        if result.get("answer") and result.get("sources"):
            print("\nSources:")
            seen: set[tuple[str, int, str]] = set()
            for source in result["sources"]:
                key = (source.arxiv_id, source.page, source.section)
                if key in seen:
                    continue
                seen.add(key)
                location = f"Page {source.page}" if source.page else "Page unavailable"
                if source.section:
                    location += f" — {source.section}"
                chunk = f" — {source.chunk_id}" if source.chunk_id else ""
                print(f"- {location}{chunk} — arXiv:{source.arxiv_id}")
    if result.get("debug"):
        _print_debug(result)
    for error in result.get("errors", []):
        print(f"warning: {error}")


def _print_debug(result: dict[str, Any]) -> None:
    selected = result.get("selected_papers", [])
    paper_id = selected[0].arxiv_id if selected else result.get("context_paper_id", "none")
    print("\n[DEBUG] Retrieval diagnostics")
    print(f"  Active paper: {paper_id}")
    print(f"  Collection: {result.get('collection_name', 'paper_' + str(paper_id).replace('.', '_'))}")
    candidates = result.get("retrieved_candidates", [])
    reranked = result.get("reranked_chunks", result.get("retrieved", []))
    print(f"  Initial candidates: {len(candidates)}")
    print(f"  Reranked/final chunks: {len(reranked)}")
    for label, hits in (("initial", candidates), ("final", reranked)):
        print(f"  {label}:")
        for hit in hits:
            chunk = hit.chunk
            print(
                f"    {chunk.chunk_id} | score={hit.score:.4f} | "
                f"page={chunk.page} | section={chunk.section or 'unavailable'}"
            )
    print(f"  Grounded: {result.get('grounded', 'not evaluated')}")
    if "grounding_score" in result:
        print(f"  Grounding score: {result['grounding_score']:.4f}")


def _run(
    graph: Any,
    query: str,
    classification: str | None = None,
    max_results: int = 5,
    context_paper_id: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    debug: bool = False,
) -> dict:
    state: dict[str, Any] = {"query": query, "max_results": max_results}
    if classification:
        state["classification"] = classification
    if context_paper_id:
        state["context_paper_id"] = context_paper_id
    if conversation_history:
        state["conversation_history"] = conversation_history
    if debug:
        state["debug"] = True
    return graph.invoke(state)


def _interactive(graph: Any, initial_query: str | None = None) -> int:
    topic = initial_query or input("Research topic or arXiv paper: ").strip()
    if not topic:
        print("A topic or paper is required.")
        return 2
    result = _run(graph, topic)
    _print_result(result)
    if result.get("briefing") and result.get("selected_papers") and not result.get("errors"):
        _save_active(result)
    if result.get("errors") and not result.get("briefing"):
        return 1
    print("\nQA mode (press Enter on an empty question to exit).")
    selected = result.get("selected_papers", [])
    context_paper_id = selected[0].arxiv_id if selected else None
    conversation_history: list[dict[str, str]] = []
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        answer = _run(
            graph,
            question,
            classification="qa",
            max_results=1,
            context_paper_id=context_paper_id,
            conversation_history=conversation_history,
        )
        _print_result(answer)
        conversation_history.append(
            {"question": question, "answer": answer.get("answer", "")}
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    if args.command == "prompt":
        try:
            return _interactive(build_graph(), args.query)
        except (KeyboardInterrupt, EOFError):
            print()
            return 130
        except Exception as exc:
            print(f"error: {exc}")
            return 1
    if not args.command:
        parser().print_help()
        return 2
    if args.command == "remove":
        try:
            paper_id = extract_arxiv_id(args.query)
            if not paper_id or not is_arxiv_reference(args.query):
                print("error: provide a valid arXiv ID or URL")
                return 2
            services = build_services()
            message = services.vector.remove_paper_embeddings(paper_id)
            active = _session_store().load()
            if active and active["paper_id"] == paper_id:
                _session_store().clear()
                message += " Active paper cleared."
            print(message)
            return 0
        except Exception as exc:
            print(f"error: embedding removal failed: {exc}")
            return 1
    classification = {"search": "search", "brief": "briefing", "ask": "qa", "qa": "qa"}[args.command]
    if args.command in {"ask", "qa"} and not args.paper:
        active = _session_store().load()
        if not active:
            print("No active paper is selected. Run 'brief <arxiv_id>' or provide --paper <arxiv_id> first.")
            return 2
        args.paper = active["paper_id"]
    if (
        args.command == "brief"
        and extract_arxiv_id(args.query) is None
        and re.fullmatch(r"[\w.-]+", args.query)
        and any(char in args.query for char in ".-")
    ):
        print("error: invalid arXiv ID or URL; provide a valid arXiv reference or a research topic")
        return 2
    try:
        result = _run(
            build_graph(),
            args.query,
            classification,
            args.max_results,
            context_paper_id=args.paper,
            debug=args.debug,
        )
    except Exception as exc:
        print(f"error: {exc}")
        return 1
    _print_result(result, args.as_json)
    if args.command == "brief" and result.get("selected_papers") and not result.get("errors"):
        _save_active(result)
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())
