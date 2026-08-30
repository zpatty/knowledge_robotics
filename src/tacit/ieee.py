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

# The per-call ceiling. Confirmed, not assumed: IEEE's own distributed sample client
# (see `xploreapi.py` on the `master` branch) hardcodes `resultSetMaxCap = 200` and
# silently clamps any larger request down to it. Do not raise this — an over-large
# max_records is clamped, not rejected, which would leave gaps in the harvest that
# look like missing papers.
MAX_RECORDS = 200
MAX_RECORDS_ASSUMED = MAX_RECORDS  # backwards-compatible alias

# Deep paging is only coherent under a deterministic sort. IEEE's sample client
# defaults to sort_field=article_title/sort_order=asc and pages with start_record;
# without an explicit sort the server's ordering is unspecified between calls, and
# records can be skipped or duplicated across page boundaries. Under a 200-call/day
# budget a silently-gapped harvest is the expensive failure, so pin the sort.
SORT_FIELD = "article_title"
SORT_ORDER = "asc"

# publication_title as it appears in Xplore. The primary strings below are the ones
# actually used in the 2020 harvest on the `master` branch, whose commit log reports
# a successful full-corpus scrape of both venues — note they carry no "IEEE" or
# "IEEE/RSJ" prefix, which the prefixed variants in the first draft of this file did.
# Proceedings titles have changed wording across forty years, so treat these as the
# best-known starting point and not as settled; the probe should try the variants.
VENUES = {
    "ICRA": "International Conference on Robotics and Automation",
    "IROS": "International Conference On Intelligent Robots and Systems",
}

VENUE_TITLE_VARIANTS = {
    "ICRA": [
        "International Conference on Robotics and Automation",
        "IEEE International Conference on Robotics and Automation",
    ],
    "IROS": [
        "International Conference On Intelligent Robots and Systems",
        "IEEE/RSJ International Conference on Intelligent Robots and Systems",
        "IEEE/RSJ International Workshop on Intelligent Robots and Systems",
    ],
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
        # Pin the sort so start_record paging is stable across calls (see SORT_FIELD).
        params.setdefault("sort_field", SORT_FIELD)
        params.setdefault("sort_order", SORT_ORDER)
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
        step = max_records or MAX_RECORDS
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
