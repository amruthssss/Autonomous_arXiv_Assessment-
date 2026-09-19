"""Gemini generation with explicit context grounding."""

from __future__ import annotations

import os
import json
from collections.abc import Iterable

from app.state import ExecutiveBriefing, Paper, SearchHit


class GeminiService:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        self._client: object | None = None

    def _get_client(self) -> object:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        if self._client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise RuntimeError("google-genai is required for Gemini generation") from exc
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @staticmethod
    def _sources(hits: Iterable[SearchHit]) -> str:
        return "\n\n".join(
            (
                f"[arXiv:{hit.chunk.paper_id}, p.{hit.chunk.page}"
                f"{', ' + hit.chunk.section if hit.chunk.section else ''}] {hit.chunk.text}"
            )
            for hit in hits
        )

    def _generate(self, prompt: str) -> str:
        response = self._get_client().models.generate_content(  # type: ignore[attr-defined]
            model=self.model,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response")
        return str(text).strip()

    def _generate_briefing(self, prompt: str) -> ExecutiveBriefing:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        client = self._get_client()
        try:
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": ExecutiveBriefing,
                },
            )
        except TypeError:
            # Older google-genai versions accept only model and contents.
            response = client.models.generate_content(  # type: ignore[attr-defined]
                model=self.model, contents=prompt
            )
        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty briefing")
        try:
            raw = str(text).strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:].lstrip()
            return ExecutiveBriefing.model_validate(json.loads(raw))
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Gemini returned an invalid structured briefing") from exc

    def briefing(self, papers: list[Paper], hits: list[SearchHit], query: str) -> ExecutiveBriefing:
        metadata = "\n".join(
            (
                f"- title: {paper.title}\n"
                f"  authors: {', '.join(paper.authors)}\n"
                f"  arxiv_id: {paper.arxiv_id}\n"
                f"  published: {paper.published}\n"
                f"  link: {paper.abs_url}"
            )
            for paper in papers
        )
        prompt = (
            "You are a careful research analyst. Return JSON matching the ExecutiveBriefing "
            "schema. Copy authors, arxiv_id, published, and link from the supplied metadata. "
            "Include a concise problem_statement, non-empty limitations and "
            "follow_up_questions arrays. "
            "Write a concise briefing answering the "
            f"request: {query!r}. Use only the supplied paper metadata and excerpts. "
            "Populate sources with arxiv_id, page, and section for every cited excerpt. "
            "Cite claims inline as [arXiv:id, p.N]. If evidence is absent, say so.\n\n"
            f"PAPERS:\n{metadata}\n\nEXCERPTS:\n{self._sources(hits)}"
        )
        return self._generate_briefing(prompt)

    def answer(
        self,
        query: str,
        hits: list[SearchHit],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        prior_turns = ""
        if history:
            prior_turns = "\n\nCONVERSATION HISTORY:\n" + "\n".join(
                f"Q: {turn.get('question', '')}\nA: {turn.get('answer', '')}"
                for turn in history[-5:]
            )
        prompt = (
            "Answer the question using only the excerpts below. Every factual claim must "
            "have an inline citation [arXiv:id, p.N]. Do not invent details; say that the "
            f"evidence is insufficient when necessary.\nQUESTION: {query}{prior_turns}\n\n"
            f"EXCERPTS:\n{self._sources(hits)}"
        )
        return self._generate(prompt)
