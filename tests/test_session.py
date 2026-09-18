from pathlib import Path

from app.services.session import ActivePaperStore
from app.services.vector_store import ChromaVectorStore


def test_active_paper_store_round_trip_and_clear(tmp_path: Path):
    store = ActivePaperStore(tmp_path / "session.json")
    store.save("1706.03762", "paper_1706_03762", "Attention Is All You Need")
    assert store.load()["paper_id"] == "1706.03762"
    store.clear()
    assert store.load() is None


class _Collection:
    def __init__(self, name: str):
        self.name = name


class _Client:
    def __init__(self):
        self.deleted = []

    def list_collections(self):
        return [_Collection("paper_1706_03762"), _Collection("paper_other")]

    def delete_collection(self, name):
        self.deleted.append(name)


def test_remove_paper_embeddings_deletes_only_requested_collection(tmp_path: Path):
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.persist_dir = tmp_path
    store._client = _Client()
    store._collections = {"paper_1706_03762": object(), "paper_other": object()}

    message = store.remove_paper_embeddings("1706.03762")

    assert "paper_1706_03762" in message
    assert store._client.deleted == ["paper_1706_03762"]
    assert "paper_other" in store._collections


def test_remove_missing_embeddings_is_graceful(tmp_path: Path):
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    store.persist_dir = tmp_path
    store._client = _Client()
    store._collections = {}

    assert "No local embeddings found" in store.remove_paper_embeddings("404.00000")
