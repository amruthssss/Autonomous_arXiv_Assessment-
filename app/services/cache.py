"""Small, atomic JSON cache with stable keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class JsonCache:
    """A process-safe-enough cache for CLI workloads.

    Values are JSON encoded and written atomically. A malformed or stale entry
    is treated as a miss instead of making a research run fail.
    """

    def __init__(self, root: Path, namespace: str) -> None:
        self.root = root
        self.namespace = namespace
        self.path = root / f"{namespace}.json"
        self.root.mkdir(parents=True, exist_ok=True)

    def _key(self, value: str) -> str:
        return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get(self._key(key))
        except (OSError, ValueError, TypeError):
            return None

    def set(self, key: str, value: Any) -> None:
        data: dict[str, Any] = {}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, ValueError, TypeError):
                data = {}
        data[self._key(key)] = value
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)

