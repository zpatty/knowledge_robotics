"""OpenAlex client.

Role split (see docs/03-corpus.md): OpenAlex was to carry the bulk harvest because
its budget was generous, with the scarce IEEE budget reserved for what only IEEE
has. Both halves of that premise failed on first contact with the live APIs — see
docs/08-phase0-findings.md. OpenAlex is now metered (1,000 credits/day free) and
its ICRA/IROS venue linkage covers under 10% of the expected corpus. Treat this
client as provisional until F3 in that document is settled.
"""
from __future__ import annotations

import logging
from typing import Iterator

from .budget import DailyBudget
from .config import optional, require
from .http import ApiClient, ApiError

log = logging.getLogger(__name__)

BASE = "https://api.openalex.org"

# Only the fields the study needs. select= materially reduces payload size and
# response time on a 60k-work pull.
WORK_FIELDS = ",".join([
    "id", "doi", "title", "publication_year", "publication_date", "type",
    "authorships", "primary_location", "locations", "referenced_works",
    "cited_by_count", "concepts", "topics", "abstract_inverted_index",
    "open_access", "biblio", "language",
])


# OpenAlex is metered in *credits*, not calls, and the free tier is far smaller than
# this project's first draft assumed. Verified against the live API on 2026-08-30:
#
#   x-ratelimit-limit:      1000 credits/day   (x-ratelimit-limit-usd: 0.10)
#   per-page=1              1 credit    ($0.0001)
#   per-page=200           10 credits   ($0.0010)
#
# so credits ≈ ceil(per_page / 20), minimum 1. A ~65k-work harvest at per-page=100
# is ~650 calls but ~3,250 credits — over three days of the free allowance, not the
# "comfortable afternoon" docs/03 §3.3 assumed. Budgeting by call count would have
# under-counted the true spend by 5x and blown the quota mid-harvest.
FREE_DAILY_CREDITS = 1000
CREDITS_PER_RESULTS = 20


def credits_for(per_page: int) -> int:
    """Credits a single call costs, as a function of page size."""
    return max(1, -(-int(per_page) // CREDITS_PER_RESULTS))


class OpenAlex:
    def __init__(self, *, dry_run: bool = False, daily_limit: int = FREE_DAILY_CREDITS):
        self.client = ApiClient(
            "openalex",
            DailyBudget("openalex_credits", daily_limit),
            min_interval=0.05,  # well under the 100 req/s ceiling
            dry_run=dry_run,
        )
        # The key is optional. OpenAlex's "polite pool" — authenticated only by a
        # mailto address — is free, needs no registration, and is rated far above
        # what this project spends (a full corpus pull is ~650 calls). A key raises
        # the ceiling further, but nothing here depends on having one.
        #
        # An *invalid* key is worse than none: OpenAlex answers it with a hard 401
        # rather than degrading to the polite pool. So a rejected key falls back,
        # loudly, once — otherwise a key that expires mid-harvest kills a run that
        # would have completed keyless.
        self.key = optional("OPENALEX_API_KEY")
        self.mailto = require("OPENALEX_MAILTO")
        self._key_rejected = False

    def _get(self, path: str, params: dict):
        params = dict(params)
        params.setdefault("mailto", self.mailto)
        url = f"{BASE}/{path}"
        cost = credits_for(params.get("per-page", 25))
        secret = {"api_key": self.key} if (self.key and not self._key_rejected) else None
        try:
            return self.client.get(url, params, secret_params=secret, cost=cost)
        except ApiError as exc:
            if secret is None or "401" not in str(exc):
                raise
            self._key_rejected = True
            log.warning(
                "OPENALEX_API_KEY was rejected (%s). Continuing on the keyless polite "
                "pool, which is sufficient for this project's call volume. Fix or "
                "unset the key to silence this.",
                str(exc).split(":")[-1].strip()[:80],
            )
            return self.client.get(url, params, secret_params=None, cost=cost)

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
