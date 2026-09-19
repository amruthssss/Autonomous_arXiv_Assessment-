"""Command-line interface for the grounded arXiv research assistant."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.graph import build_graph
from app.nodes import index_papers
from app.services.arxiv import ArxivAPIError, extract_arxiv_id, is_arxiv_reference
from app.services.config import build_services
from app.services.session import ActivePaperStore
from app.state import ExecutiveBriefing, Paper


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
    print("SEARCH RESULTS")
    print("=" * 60)

    print(f"\nQuery: {query}")
    status = result.get("search_status")
    if status == "success":
        print("\nSEARCH_STATUS: SUCCESS")
    elif status == "no_results":
        print("\nSEARCH_STATUS: NO_RESULTS")
    elif status == "arxiv_unavailable":
        print("\nSEARCH_STATUS: ARXIV_UNAVAILABLE")
    print(f"\nFound {len(papers)} paper(s).")
    for index, paper in enumerate(papers, start=1):
        score = scores[index - 1] if index - 1 < len(scores) else None
        final = f"{score.total:.3f}" if score is not None else "Not available"
        print(f"\n[{index}] {_display(paper.title)}")
        print(f"    arXiv: {paper.arxiv_id}")
        print(f"    Relevance: {final}")
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
        print("\nBest match automatically selected: [1]")
        print("\nNext command:\nfetch")
        if len(papers) > 1:
            print("\nWant another paper?\nselect <number>")


def _print_result(result: dict[str, Any], as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(result, default=_jsonable, indent=2))
        return
    if result.get("classification") == "search":
        if result.get("search_status") in {"arxiv_unavailable", "error"}:
            print(
                "SEARCH_STATUS: ARXIV_UNAVAILABLE"
                if result.get("search_status") == "arxiv_unavailable"
                else "SEARCH_STATUS: ERROR"
            )
            if result.get("search_status") == "arxiv_unavailable":
                print("The official arXiv API rejected or could not serve the request.")
                print("No paper-result conclusion was made.")
            for error in result.get("errors", []):
                print(f"ERROR: {error}")
            if result.get("debug"):
                _print_debug(result)
            return
        _print_topic_search(result)
        if result.get("debug"):
            _print_debug(result)
        for error in result.get("errors", []):
            print(f"warning: {error}")
        return
    briefing = result.get("briefing")
    if result.get("search_status") == "no_results":
        print("SEARCH_STATUS: NO_RESULTS")
    elif result.get("search_status") == "arxiv_unavailable":
        print("SEARCH_STATUS: ARXIV_UNAVAILABLE")
    if result.get("paper_status"):
        print(f"PAPER_STATUS: {str(result['paper_status']).upper()}")
    if result.get("fetch_status"):
        print(f"FETCH_STATUS: {str(result['fetch_status']).upper()}")
    if result.get("parse_status"):
        print(f"PARSE_STATUS: {str(result['parse_status']).upper()}")
    if result.get("index_status"):
        print(f"INDEX_STATUS: {str(result['index_status']).upper()}")
    if isinstance(briefing, ExecutiveBriefing):
        print(f"# {briefing.title}")
        print(f"\nAuthors: {', '.join(briefing.authors)}")
        print(f"arXiv ID: {briefing.arxiv_id}")
        print(f"Published: {briefing.published}")
        print(f"Link: {briefing.link}")
        print(f"\nWhy it matters:\n{briefing.executive_summary}")
        print(f"\nProblem statement:\n{briefing.problem_statement}")
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
        if isinstance(briefing, str) and "couldn't find enough information" in briefing.lower():
            print("BRIEF_STATUS: INSUFFICIENT_EVIDENCE")
        if result.get("answer") and result.get("grounded") is False:
            print("QA_STATUS: INSUFFICIENT_EVIDENCE")
            print("Gemini was not used because the retrieved evidence was insufficient.")
        elif result.get("answer") and result.get("grounded") is True:
            print("QA_STATUS: GROUNDED")
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
                print(f"- {location}")
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
    if result.get("paper_index_status"):
        print(f"  Index status: {result['paper_index_status']}")
    if result.get("index_diagnostic"):
        print(f"  Index: {result['index_diagnostic']}")
    hits = result.get("retrieved", [])
    print(f"  Retrieved chunks: {len(hits)}")
    for hit in hits:
        chunk = hit.chunk
        print(
            f"    {chunk.chunk_id} | score={hit.score:.4f} | "
            f"page={chunk.page} | section={chunk.section or 'unavailable'}"
        )
    print(f"  Grounded: {result.get('grounded', 'not evaluated')}")
    if "grounding_score" in result:
        print(f"  Grounding score: {result['grounding_score']:.4f}")


def _briefing_markdown(briefing: ExecutiveBriefing) -> str:
    lines = [
        f"# {briefing.title}",
        "",
        f"- **arXiv ID:** {briefing.arxiv_id}",
        f"- **Authors:** {', '.join(briefing.authors) or 'Not available'}",
        f"- **Published:** {briefing.published or 'Not available'}",
        f"- **Link:** {briefing.link or 'Not available'}",
        f"- **Exported:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Executive summary",
        "",
        briefing.executive_summary,
        "",
        "## Problem statement",
        "",
        briefing.problem_statement,
        "",
        "## Method",
        "",
        briefing.methods,
        "",
        "## Key findings",
        "",
    ]
    lines.extend(f"- {finding}" for finding in briefing.key_findings)
    lines.extend(["", "## Results", ""])
    lines.extend(f"- {result}" for result in briefing.results)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in briefing.limitations)
    lines.extend(["", "## Follow-up questions", ""])
    lines.extend(f"- {question}" for question in briefing.follow_up_questions)
    lines.extend(["", "## Sources", ""])
    seen: set[tuple[str, int, str]] = set()
    for source in briefing.sources:
        key = (source.arxiv_id, source.page, source.section)
        if key in seen:
            continue
        seen.add(key)
        location = f"Page {source.page}" if source.page else "Page unavailable"
        if source.section:
            location += f" — {source.section}"
        lines.append(f"- {location} — arXiv:{source.arxiv_id}")
    return "\n".join(lines).rstrip() + "\n"


def _export_briefing(briefing: ExecutiveBriefing, target: str | None = None) -> Path:
    path = Path(target or f"briefing_{briefing.arxiv_id.replace('.', '_')}.md").expanduser()
    if path.suffix.lower() != ".md":
        path = path.with_suffix(".md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_briefing_markdown(briefing), encoding="utf-8")
    return path


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


def _interactive_help() -> None:
    print(
        "\nAvailable commands\n"
        "--------------------------------------------------\n"
        "search <topic>       Search arXiv and select the best result\n"
        "select <number>      Select another latest search result\n"
        "fetch                Download, parse, chunk, embed, and index\n"
        "summarise            Generate the executive briefing\n"
        "export [file]        Export the briefing to Markdown\n"
        "ask <question>       Ask about the current paper\n"
        "switch <number>      Switch between indexed papers\n"
        "papers               Show indexed papers\n"
        "current              Show the current paper and status\n"
        "status               Show active paper pipeline status\n"
        "help                 Show this help\n"
        "exit                 Save the session and exit\n"
        "\nTypical workflow:\n"
        "  search transformer architecture\n"
        "  fetch\n"
        "  summarise\n"
        "  ask What is the main contribution?\n"
        "--------------------------------------------------"
    )


def _interactive_papers(services: Any, current_id: str | None) -> list[dict[str, object]]:
    papers = services.vector.indexed_papers()
    print("\nIndexed papers")
    print("-" * 50)
    if not papers:
        print("No indexed papers.")
    for index, paper in enumerate(papers, start=1):
        marker = " (current)" if paper["paper_id"] == current_id else ""
        print(
            f"[{index}] {paper['title'] or 'Untitled'}{marker}\n"
            f"    ID: {paper['paper_id']}\n"
            f"    Status: {str(paper['status']).upper()} | Chunks: {paper['chunks']}"
        )
    return papers


def _interactive_current(
    services: Any, paper: Paper | None, briefing: ExecutiveBriefing | None
) -> None:
    if paper is None:
        print("No current paper selected.")
        return
    index_status, diagnostic = services.vector.inspect_index(paper.arxiv_id)
    print("\n" + "=" * 50)
    print("CURRENT PAPER")
    print("=" * 50)
    print(f"\nTitle:\n{paper.title}")
    print(f"\narXiv ID:\n{paper.arxiv_id}")
    print(f"\nAuthors:\n{_display(', '.join(paper.authors))}")
    print(f"\nPublished:\n{_display(paper.published)}")
    print(f"\nCategories:\n{_display(', '.join(paper.categories))}")
    print(f"\nPDF:\n{services.pdf.path_for(paper)}")
    print(f"\nPDF Status:\n{'AVAILABLE' if services.pdf.path_for(paper).exists() else 'NOT AVAILABLE LOCALLY'}")
    print("\nIndex Status:")
    print(f"{str(index_status).upper()}")
    if index_status == "indexed":
        print(diagnostic)
    print(f"\nEmbedding Model:\n{services.vector.embeddings.model_name}")
    print(f"\nChroma Collection:\n{services.vector.collection_name(paper.arxiv_id)}")
    print(f"\nBriefing:\n{'AVAILABLE' if briefing else 'NOT GENERATED'}")
    print("\n" + "=" * 50)


def _interactive_status(
    services: Any, paper: Paper | None, briefing: ExecutiveBriefing | None
) -> None:
    print("\nSTATUS")
    print("-" * 50)
    if paper is None:
        print("Active paper: None")
        print(f"Indexed papers: {len(services.vector.indexed_papers())}")
        return
    index_status, diagnostic = services.vector.inspect_index(paper.arxiv_id)
    indexed = next(
        (
            item
            for item in services.vector.indexed_papers()
            if item["paper_id"] == paper.arxiv_id
        ),
        {},
    )
    pdf_path = services.pdf.path_for(paper)
    print(f"Active paper: {paper.arxiv_id}")
    print(f"Title: {paper.title}")
    print(f"PDF: {'AVAILABLE' if pdf_path.exists() else 'NOT AVAILABLE'}")
    print(f"Index: {str(index_status).upper()}")
    print(f"Chunks: {indexed.get('chunks', 0)}")
    print(f"Embedding model: {services.vector.embeddings.model_name}")
    print(f"Collection: {services.vector.collection_name(paper.arxiv_id)}")
    print(f"Briefing: {'AVAILABLE' if briefing else 'NOT GENERATED'}")
    if diagnostic:
        print(f"Details: {diagnostic}")


def _interactive_session() -> int:
    services = build_services()
    graph = build_graph(services)
    latest_results: list[Paper] = []
    current: Paper | None = None
    briefing: ExecutiveBriefing | None = None
    history: list[dict[str, str]] = []
    saved = _session_store().load()
    if saved:
        try:
            resolved = services.arxiv.resolve(saved["paper_id"])
            current = resolved
        except Exception:
            current = None

    print("=" * 50)
    print("        Autonomous arXiv Paper Agent")
    print("=" * 50)
    print(f"\nIndexed papers: {len(services.vector.indexed_papers())}")
    if current:
        index_status, _ = services.vector.inspect_index(current.arxiv_id)
        print(
            f"\nCurrent paper:\n{current.title}\n"
            f"arXiv: {current.arxiv_id}\n"
            f"Status: {'Indexed' if index_status == 'indexed' else 'Not indexed'}"
        )
    else:
        print("\nCurrent paper: None")
    print("\nYou can:")
    print("1. Search for a research topic")
    print("2. Open a paper using an arXiv ID or URL")
    print("\nType `help` to see commands.")
    print("Type `exit` to quit.")

    while True:
        try:
            raw = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving session...\nGoodbye.")
            return 0
        if not raw:
            continue
        parts = raw.split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) == 2 else ""
        if command == "exit":
            print("Saving session...\nGoodbye.")
            return 0
        if command == "help":
            _interactive_help()
            continue
        if command == "search":
            topic = argument or input("\nEnter research topic:\n> ").strip()
            if not topic:
                print("A research topic is required.")
                continue
            result = _run(graph, topic, classification="search")
            latest_results = result.get("papers", [])
            _print_topic_search(result)
            if latest_results and result.get("search_status") == "success":
                current = latest_results[0]
                briefing = None
                history = []
                _save_active({"selected_papers": [current]})
            continue
        if command == "select":
            if not latest_results and not extract_arxiv_id(argument):
                print("No search results. Run 'search' first.")
                continue
            if latest_results:
                print("\nAvailable papers:")
                for number, paper in enumerate(latest_results, start=1):
                    print(f"\n[{number}] {paper.title}\n    {paper.arxiv_id}")
            choice = argument or input("\nSelect paper number:\n> ").strip()
            try:
                if extract_arxiv_id(choice):
                    selected = next(
                        (
                            paper
                            for paper in latest_results
                            if paper.arxiv_id == extract_arxiv_id(choice)
                        ),
                        None,
                    )
                    if selected is None:
                        selected = services.arxiv.resolve(extract_arxiv_id(choice))
                else:
                    selected = latest_results[int(choice) - 1]
            except (ValueError, IndexError, ArxivAPIError) as exc:
                if isinstance(exc, ArxivAPIError):
                    print(f"Could not select paper: {exc}")
                    continue
                print("Invalid paper number.")
                continue
            current = selected
            briefing = None
            history = []
            _save_active({"selected_papers": [current]})
            index_status, diagnostic = services.vector.inspect_index(current.arxiv_id)
            next_command = "summarise" if index_status == "indexed" else "fetch"
            print(
                f"\nSelected paper:\n{current.title}\n"
                f"arXiv ID: {current.arxiv_id}\n"
                f"Status: {'Already indexed' if index_status == 'indexed' else 'Not indexed'}\n\n"
                f"Next command:\n{next_command}"
            )
            if index_status == "indexed":
                print(
                    f"\nINDEX_STATUS: READY\n{diagnostic}\n"
                    "You can now run:\nsummarise\nask <question>"
                )
            continue
        if command in {
            "fetch",
            "summarise",
            "export",
            "ask",
            "switch",
            "papers",
            "current",
            "status",
        }:
            if command == "papers":
                _interactive_papers(services, current.arxiv_id if current else None)
                continue
            if command == "status":
                _interactive_status(services, current, briefing)
                continue
            if command == "switch":
                papers = _interactive_papers(services, current.arxiv_id if current else None)
                if not papers:
                    continue
                choice = argument or input("\nSelect paper number or arXiv ID:\n> ").strip()
                try:
                    if extract_arxiv_id(choice):
                        selected = next(
                            paper
                            for paper in papers
                            if paper["paper_id"] == extract_arxiv_id(choice)
                        )
                    else:
                        selected = papers[int(choice) - 1]
                    resolved = services.arxiv.resolve(str(selected["paper_id"]))
                except (ValueError, IndexError, ArxivAPIError) as exc:
                    print(f"Could not switch paper: {exc}")
                    continue
                current, briefing, history = resolved, None, []
                _save_active({"selected_papers": [current]})
                print(
                    f"\nCURRENT PAPER:\n{current.title}\n"
                    f"arXiv ID: {current.arxiv_id}\n\nNext command:\nask <question>"
                )
                continue
            if command == "current":
                _interactive_current(services, current, briefing)
                continue
            if command == "export":
                if briefing is None:
                    print("No briefing is available. Run 'summarise' first.")
                    continue
                try:
                    path = _export_briefing(briefing, argument or None)
                except OSError as exc:
                    print(f"Could not export briefing: {exc}")
                    continue
                print(f"Briefing exported to: {path}")
                continue
            if current is None:
                print("No current paper selected. Use 'search'/'select' or enter an arXiv ID/URL.")
                continue
            if command == "fetch":
                if argument:
                    direct_id = extract_arxiv_id(argument)
                    if not direct_id or not is_arxiv_reference(argument):
                        print("Provide a valid arXiv ID or URL.")
                        continue
                    try:
                        current = services.arxiv.resolve(direct_id)
                    except ArxivAPIError as exc:
                        print(f"Could not fetch paper: {exc}")
                        continue
                    briefing, history = None, []
                    _save_active({"selected_papers": [current]})
                indexed = index_papers(
                    {
                        "classification": "briefing",
                        "search_status": "success",
                        "selected_papers": [current],
                    },
                    services,
                )
                indexed_summary = next(
                    (
                        item
                        for item in services.vector.indexed_papers()
                        if item["paper_id"] == current.arxiv_id
                    ),
                    {},
                )
                print(f"\n{'=' * 50}\nFETCH PAPER\n{'=' * 50}")
                print(f"\nPaper:\n{current.title}\n\narXiv ID:\n{current.arxiv_id}")
                fetch_status = str(indexed.get("fetch_status", "failed")).upper()
                parse_status = str(indexed.get("parse_status", "failed")).upper()
                index_ready = indexed.get("index_status") == "ready"
                chunk_count = len(indexed.get("chunks", [])) or indexed_summary.get("chunks", 0)
                print(f"\nPDF          {'[OK] Ready' if fetch_status == 'SUCCESS' else '[FAILED] ' + fetch_status.title()}")
                print(f"Parsing      {'[OK] Success' if parse_status == 'SUCCESS' else '[FAILED] ' + parse_status.title()}")
                print(f"Chunks       {'[OK] ' + str(chunk_count) if chunk_count else '[FAILED] None created'}")
                print(f"Embeddings   {'[OK] Ready' if index_ready else '[FAILED] Not ready'}")
                print(f"Index        {'[OK] Ready' if index_ready else '[FAILED] Not ready'}")
                diagnostic = indexed.get("index_diagnostic", "")
                if diagnostic:
                    print(f"\n{diagnostic}")
                if not index_ready and fetch_status == "SUCCESS" and parse_status == "SUCCESS":
                    print("\nThe paper was downloaded and parsed successfully, but indexing failed.")
                    print("Your paper is safe to retry; no new paper selection is needed.")
                    print("\nNext command:\nfetch")
                if indexed.get("index_status") == "ready":
                    print("\nPaper is ready for research.\n\nNext command:\nsummarise")
                continue
            if command == "summarise":
                result = _run(
                    graph,
                    current.arxiv_id,
                    classification="briefing",
                    context_paper_id=current.arxiv_id,
                )
                _print_result(result)
                if isinstance(result.get("briefing"), ExecutiveBriefing):
                    briefing = result["briefing"]
                    print(
                        "\nBriefing generated.\n"
                        "You can now ask questions:\n"
                        "ask <question>\n\n"
                        "Example:\n"
                        "ask What is the main contribution?"
                    )
                continue
            debug = False
            if argument.endswith(" --debug"):
                argument = argument[:-8].rstrip()
                debug = True
            question = argument or input("\nQuestion:\n> ").strip()
            if not question:
                continue
            result = _run(
                graph,
                question,
                classification="qa",
                context_paper_id=current.arxiv_id,
                conversation_history=history,
                debug=debug,
            )
            _print_result(result)
            history.append({"question": question, "answer": result.get("answer", "") or ""})
            history = history[-5:]
            print("\nYou can ask another question:\nask <question>")
            continue

        direct_id = extract_arxiv_id(raw)
        if direct_id and is_arxiv_reference(raw):
            result = _run(graph, raw, classification="paper")
            selected = result.get("selected_papers", [])
            if selected:
                current, briefing, history = selected[0], None, []
                _save_active(result)
                index_status, _ = services.vector.inspect_index(current.arxiv_id)
                print(
                    f"\nSelected paper:\n{current.title}\n"
                    f"arXiv ID: {current.arxiv_id}\n"
                    f"Status: {'Already indexed' if index_status == 'indexed' else 'Not indexed'}\n\n"
                    f"Next command:\n{'summarise' if index_status == 'indexed' else 'fetch'}"
                )
            else:
                _print_result(result)
            continue
        print(
            f"Unknown command: {command}\n\n"
            "Type `help` to see available commands.\n"
            "Typical workflow: search <topic> -> fetch -> summarise -> ask <question>"
        )


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parser().parse_args(argv)
    if args.command == "prompt":
        try:
            return _interactive(build_graph(build_services()), args.query)
        except (KeyboardInterrupt, EOFError):
            print()
            return 130
        except Exception as exc:
            print(f"error: {exc}")
            return 1
    if not args.command:
        try:
            return _interactive_session()
        except (KeyboardInterrupt, EOFError):
            print("\nSaving session...\nGoodbye.")
            return 0
        except Exception as exc:
            print(f"error: {exc}")
            return 1
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
            build_graph(build_services(debug=args.debug)),
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
