"""OpenAlex client — the bulk workhorse.

Role split (see docs/03-corpus.md): OpenAlex carries the bulk harvest because its
budget is generous, and the scarce IEEE budget is reserved for what only IEEE has.
"""
from __future__ import annotations

from typing import Iterator

from .budget import DailyBudget
from .config import require
from .http import ApiClient

BASE = "https://api.openalex.org"

# Only the fields the study needs. select= materially reduces payload size and
# response time on a 60k-work pull.
WORK_FIELDS = ",".join([
    "id", "doi", "title", "publication_year", "publication_date", "type",
    "authorships", "primary_location", "locations", "referenced_works",
    "cited_by_count", "concepts", "topics", "abstract_inverted_index",
    "open_access", "biblio", "language",
])


class OpenAlex:
    def __init__(self, *, dry_run: bool = False, daily_limit: int = 100_000):
        self.client = ApiClient(
            "openalex",
            DailyBudget("openalex", daily_limit),
            min_interval=0.05,  # well under the 100 req/s ceiling
            dry_run=dry_run,
        )
        self.key = require("OPENALEX_API_KEY")
        self.mailto = require("OPENALEX_MAILTO")

    def _get(self, path: str, params: dict):
        params = dict(params)
        params.setdefault("mailto", self.mailto)
        return self.client.get(f"{BASE}/{path}", params, secret_params={"api_key": self.key})

    def find_sources(self, query: str) -> list[dict]:
        """Locate venue source records by name. Conference series are often split
        across several source records; inspect the results rather than trusting the
        first hit."""
        data = self._get("sources", {"search": query, "per-page": 25})
        return (data or {}).get("results", [])

    def count_works(self, filters: str) -> int:
        """One call, returns meta.count. Use this to size a pull before committing."""
        data = self._get("works", {"filter": filters, "per-page": 1})
        return ((data or {}).get("meta") or {}).get("count", 0)

    def iter_works(self, filters: str, *, per_page: int = 100) -> Iterator[dict]:
        """Cursor-paged work iteration. Cursor paging is required past 10k results."""
        cursor = "*"
        while cursor:
            data = self._get("works", {
                "filter": filters,
                "per-page": per_page,
                "cursor": cursor,
                "select": WORK_FIELDS,
            })
            if data is None:  # dry run
                return
            yield from data.get("results", [])
            cursor = (data.get("meta") or {}).get("next_cursor")


def decode_abstract(inverted: dict | None) -> str | None:
    """OpenAlex stores abstracts as an inverted index; rebuild the text."""
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        positions.extend((i, word) for i in idxs)
    if not positions:
        return None
    positions.sort()
    return " ".join(word for _, word in positions)
