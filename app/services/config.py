"""Application configuration and dependency construction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from app.services.arxiv import ArxivService
from app.services.embeddings import EmbeddingService
from app.services.gemini import GeminiService
from app.services.pdf import PdfService
from app.services.vector_store import ChromaVectorStore
from app.services.cache import JsonCache


@dataclass
class Services:
    arxiv: ArxivService
    pdf: PdfService
    vector: ChromaVectorStore
    gemini: GeminiService
    grounding_threshold: float = 0.20
    briefing_cache: JsonCache | None = None


def build_services(root: Path | None = None) -> Services:
    project_root = root or Path(__file__).resolve().parents[2]
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None
    if load_dotenv:
        load_dotenv(project_root / ".env", override=False)
    data = project_root / "data"
    embeddings = EmbeddingService(os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    try:
        threshold = max(
            0.0,
            min(
                1.0,
                float(
                    os.getenv(
                        "GROUNDING_THRESHOLD",
                        os.getenv("QA_GROUNDING_THRESHOLD", "0.20"),
                    )
                ),
            ),
        )
    except ValueError:
        threshold = 0.20
    return Services(
        arxiv=ArxivService(data),
        pdf=PdfService(data / "pdfs"),
        vector=ChromaVectorStore(data / "chroma", embeddings),
        gemini=GeminiService(),
        grounding_threshold=threshold,
        briefing_cache=JsonCache(data, "briefings"),
    )
