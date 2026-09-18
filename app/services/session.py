"""Small local session store for the active paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ActivePaperStore:
    """Persist only the metadata needed to resolve the current paper."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, str] | None:
        if not self.path.exists():
            return None
        try:
            value: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict) or not isinstance(value.get("paper_id"), str):
            return None
        return {
            "paper_id": value["paper_id"],
            "collection_name": str(value.get("collection_name", "")),
            "title": str(value.get("title", "")),
            "abs_url": str(value.get("abs_url", "")),
        }

    def save(self, paper_id: str, collection_name: str, title: str = "", abs_url: str = "") -> None:
        payload = {
            "paper_id": paper_id,
            "collection_name": collection_name,
            "title": title,
            "abs_url": abs_url,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)
