"""Persistent ChromaDB storage for paper chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from collections import defaultdict

from app.services.embeddings import EmbeddingService
from app.state import DocumentChunk, SearchHit


class ChromaVectorStore:
    def __init__(self, persist_dir: Path, embeddings: EmbeddingService) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings
        self._client: object | None = None
        self._collections: dict[str, object] = {}

    @staticmethod
    def collection_name(paper_id: str) -> str:
        safe_id = "".join(char if char.isalnum() else "_" for char in paper_id)
        return f"paper_{safe_id}"

    def _get_collection(self, paper_id: str) -> object:
        name = self.collection_name(paper_id)
        if name not in self._collections:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("chromadb is required for local RAG") from exc
            if self._client is None:
                self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def inspect_index(self, paper_id: str) -> tuple[str, str]:
        """Validate an existing paper collection without creating one."""

        name = self.collection_name(paper_id)
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required for local RAG") from exc
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        names = {
            getattr(collection, "name", collection)
            for collection in self._client.list_collections()
        }
        if name not in names:
            return "not_indexed", f"No Chroma collection exists for {paper_id}."
        try:
            collection = self._client.get_collection(name=name)
            data = collection.get(include=["documents", "metadatas", "embeddings"])
        except Exception as exc:
            self._collections.pop(name, None)
            return "corrupted", f"Indexed collection {name} could not be read: {exc}"
        ids = data.get("ids", [])
        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []
        embeddings = data.get("embeddings")
        if not ids or not documents or not metadatas or embeddings is None:
            return "corrupted", f"Indexed collection {name} is empty or missing embeddings."
        if len(ids) != len(documents) or len(ids) != len(metadatas):
            return "corrupted", f"Indexed collection {name} has inconsistent records."
        for document, metadata in zip(documents, metadatas, strict=False):
            if not document or not isinstance(metadata, dict):
                return "corrupted", f"Indexed collection {name} has invalid chunk data."
            if str(metadata.get("paper_id", "")) != paper_id:
                return "corrupted", f"Indexed collection {name} contains another paper."
        self._collections[name] = collection
        return "indexed", (
            f"Paper already indexed locally. Using existing indexed evidence "
            f"from {name} ({len(ids)} chunks)."
        )

    def delete_collection(self, paper_id: str) -> None:
        name = self.collection_name(paper_id)
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._client.delete_collection(name=name)
        self._collections.pop(name, None)

    def add(self, chunks: Iterable[DocumentChunk]) -> None:
        values = list(chunks)
        if not values:
            return
        groups: dict[str, list[DocumentChunk]] = defaultdict(list)
        for chunk in values:
            groups[chunk.paper_id].append(chunk)
        for paper_id, paper_chunks in groups.items():
            collection = self._get_collection(paper_id)
            existing = collection.get(  # type: ignore[attr-defined]
                ids=[chunk.chunk_id for chunk in paper_chunks],
                include=[],
            ).get("ids", [])
            if set(existing) == {chunk.chunk_id for chunk in paper_chunks}:
                continue
            collection.upsert(  # type: ignore[attr-defined]
                ids=[chunk.chunk_id for chunk in paper_chunks],
                documents=[chunk.text for chunk in paper_chunks],
                embeddings=self.embeddings.encode([chunk.text for chunk in paper_chunks]),
                metadatas=[
                    {
                        "paper_id": chunk.paper_id,
                        "title": chunk.title,
                        "page": chunk.page,
                        "start": chunk.start,
                        "end": chunk.end,
                        "section": chunk.section,
                    }
                    for chunk in paper_chunks
                ],
            )

    def query(self, text: str, paper_id: str, n_results: int = 6) -> list[SearchHit]:
        collection = self._get_collection(paper_id)
        result = collection.query(  # type: ignore[attr-defined]
            query_embeddings=[self.embeddings.embed(text)],
            n_results=n_results,
            where={"paper_id": paper_id},
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        hits: list[SearchHit] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            metadata = metadata or {}
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                paper_id=str(metadata.get("paper_id", "")),
                title=str(metadata.get("title", "")),
                text=document,
                page=int(metadata.get("page", 0)),
                start=int(metadata.get("start", 0)),
                end=int(metadata.get("end", 0)),
                section=str(metadata.get("section", "")),
            )
            hits.append(SearchHit(chunk=chunk, score=1.0 - float(distance)))
        return hits

    def remove_paper_embeddings(self, paper_id: str) -> str:
        """Delete only the deterministic collection belonging to one paper."""

        name = self.collection_name(paper_id)
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        collection_names = {
            getattr(collection, "name", collection)
            for collection in self._client.list_collections()
        }
        if name not in collection_names:
            return f"No local embeddings found for {paper_id}."
        self.delete_collection(paper_id)
        return f"Removed local embeddings for {paper_id} ({name})."

    def indexed_papers(self) -> list[dict[str, object]]:
        """Return locally indexed paper summaries discovered from Chroma."""

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required for local RAG") from exc
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        papers: list[dict[str, object]] = []
        for collection in self._client.list_collections():
            name = str(getattr(collection, "name", collection))
            if not name.startswith("paper_"):
                continue
            try:
                value = self._client.get_collection(name=name).get(
                    include=["documents", "metadatas"]
                )
                metadata = (value.get("metadatas") or [{}])[0] or {}
                papers.append(
                    {
                        "paper_id": str(metadata.get("paper_id", name.removeprefix("paper_"))),
                        "title": str(metadata.get("title", "")),
                        "collection_name": name,
                        "chunks": len(value.get("ids", [])),
                        "status": "ready",
                    }
                )
            except Exception:
                continue
        return sorted(papers, key=lambda item: str(item["paper_id"]))
