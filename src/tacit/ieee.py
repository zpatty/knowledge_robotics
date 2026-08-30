"""IEEE Xplore client — scarce, authoritative, used surgically.

200 calls/day. At a (to-be-verified) max of 200 records per call that is 40k
records/day in the best case, so a full ~65k-paper pull is a multi-day operation
with no room for waste. Every call is cached and budget-guarded, and the intended
use is NOT bulk harvest but:

  1. authoritative venue/year counts, to audit OpenAlex completeness
  2. abstracts for records where OpenAlex has none
  3. IEEE controlled index terms
  4. supplementary-material / multimedia flags, if the API exposes them

Field availability is unverified — run scripts/probe_ieee.py first. It spends ~8
calls and determines the whole harvest design.
"""
from __future__ import annotations

from typing import Any

from .budget import DailyBudget
from .config import require
from .http import ApiClient

BASE = "https://ieeexploreapi.ieee.org/api/v1/search/articles"

DAILY_CALL_LIMIT = 200
CALLS_PER_SECOND = 10

# To verify in the probe: the documented per-call ceiling. Do not raise this on a
# guess — an over-large max_records may be silently truncated, which would leave
# gaps in the harvest that look like missing papers.
MAX_RECORDS_ASSUMED = 200

VENUES = {
    # publication_title as it appears in Xplore. Verify: proceedings titles have
    # changed wording across 40 years and may need several variants per venue.
    "ICRA": "IEEE International Conference on Robotics and Automation",
    "IROS": "IEEE/RSJ International Conference on Intelligent Robots and Systems",
}


class IEEEXplore:
    def __init__(self, *, dry_run: bool = False, reserve: int = 0):
        """reserve keeps N calls of the daily budget unspent, so an automated
        harvest cannot consume the allowance an interactive probe needs."""
        limit = DAILY_CALL_LIMIT - reserve
        self.client = ApiClient(
            "ieee",
            DailyBudget("ieee", limit),
            min_interval=1.0 / CALLS_PER_SECOND,
            dry_run=dry_run,
        )
        self.key = require("IEEE_API_KEY")

    @property
    def remaining_today(self) -> int:
        return self.client.budget.remaining()

    def search(self, **params: Any) -> dict:
        params.setdefault("format", "json")
        return self.client.get(BASE, params, secret_params={"apikey": self.key})

    def count(self, publication_title: str, year: int) -> int | None:
        """One call for a venue-year total. This is the cheapest useful IEEE query
        and the basis of the completeness audit against OpenAlex."""
        data = self.search(
            publication_title=publication_title,
            publication_year=str(year),
            max_records=1,
        )
        return None if data is None else data.get("total_records")

    def fetch_year(self, publication_title: str, year: int, *, max_records: int | None = None):
        """Page one venue-year. Yields (start_record, payload) so a partial harvest
        is resumable at the exact record it stopped on."""
        step = max_records or MAX_RECORDS_ASSUMED
        start = 1
        total = None
        while total is None or start <= total:
            data = self.search(
                publication_title=publication_title,
                publication_year=str(year),
                start_record=start,
                max_records=step,
            )
            if data is None:  # dry run
                return
            total = data.get("total_records", 0)
            yield start, data
            got = len(data.get("articles", []))
            if got == 0:
                break
            start += got
