"""Content-addressed HTTP response cache.

Cache-first is not an optimization here, it is the budget policy: a request that
has been made once must never be paid for twice. Keys are the full request URL
with credentials stripped, so a rotated API key does not invalidate the cache.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .config import CACHE

_SECRET_PARAM = re.compile(r"([?&])(apikey|api_key)=[^&]*", re.IGNORECASE)


def cache_key(url: str) -> str:
    scrubbed = _SECRET_PARAM.sub(r"\1\2=REDACTED", url)
    return hashlib.sha256(scrubbed.encode()).hexdigest()


class ResponseCache:
    def __init__(self, namespace: str, root: Path | None = None):
        self.root = (root or CACHE) / namespace
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Shard by first two hex chars to keep directories navigable.
        d = self.root / key[:2]
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}.json.gz"

    def get(self, url: str) -> Any | None:
        path = self._path(cache_key(url))
        if not path.exists():
            return None
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh)

    def put(self, url: str, payload: Any) -> None:
        path = self._path(cache_key(url))
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump(payload, fh)
        tmp.replace(path)

    def stats(self) -> dict:
        files = list(self.root.rglob("*.json.gz"))
        return {
            "entries": len(files),
            "bytes": sum(f.stat().st_size for f in files),
        }
