"""Local sentence-transformer embeddings."""

from __future__ import annotations

from typing import Iterable


class EmbeddingService:
    """Lazy local embedding model, avoiding model startup for search-only runs."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for local RAG") from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        model = self._load()
        values = model.encode(list(texts), normalize_embeddings=True)  # type: ignore[attr-defined]
        return values.tolist()

    def embed(self, text: str) -> list[float]:
        return self.encode([text])[0]

