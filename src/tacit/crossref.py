"""Crossref client — Layer A and A′ for the era OpenAlex cannot reach.

Promoted from "secondary" after the Phase 0 probe (docs/08-phase0-findings.md):
OpenAlex resolves IROS to three years out of thirty-five, while Crossref carries
1995 IROS proceedings with reference edges and author affiliations. Crossref is
free, unmetered, and needs no key — only a `mailto` for the polite pool.

What it supplies, verified against live records:

  Layer A   container title, article title, DOI, year, authors with affiliation
            strings, ISBN where deposited
  Layer A′  the full `reference` array — not merely a count

What it does not supply:

  abstracts        absent from IEEE's conference deposits. The Phase 0 abstract
                   gate still needs IEEE or OpenAlex.
  complete refs    only ~47% of reference entries carry a DOI (measured on a 2002
                   IROS paper: 8 of 17). The rest are unstructured strings and
                   need matching before the citation graph is usable. This
                   directly bounds T1/T2 recall — see docs/08.

Three API traps, all found the hard way and all silent:

  1. `select` DROPS DATE FIELDS. Requesting select=published returns records with
     no `published` key at all rather than an error, so every year comes back
     empty and the corpus looks undated. Do not pass `select` when you need
     dates. (An invalid select name *is* rejected loudly; a valid-but-unsupported
     one is not.)
  2. `query.container-title` is fuzzy relevance ranking, NOT a filter. It returns
     hundreds of thousands of results for either venue and its `total-results` is
     meaningless. Never quote a count obtained from it.
  3. **CURSOR PAGING DISCARDS RELEVANCE ORDER.** This is the expensive one. A
     `query.*` search returns its best matches first only on an un-cursored
     request; `cursor=*` re-orders the whole result set by an internal key, so
     paging a fuzzy query walks essentially arbitrary records. Measured on ICRA
     2015: the first un-cursored page is 200/200 genuine ICRA, while six
     cursor-paged pages over the same query and filter return *zero*. Use
     `offset` paging with a fuzzy query; use `cursor` only with a pure filter.

The `member:263 + date` route looks like the principled alternative and does not
work either: Crossref's date filter and its own record data disagree for
backfilled proceedings. `10.1109/robot.2002.1013340` carries member 263, type
proceedings-article and published 2002, yet enumerating exactly that filter for
2002 returns 9,673 works containing only 21 `robot.*` DOIs. The records exist and
resolve individually by DOI; they are simply not reachable through the search
index. Deposit dates suggest why — IEEE re-deposited this era in 2022 and it was
indexed in 2025.

**Coverage boundary, measured** (ICRA+IROS papers recovered per year, after the
venue exclusions below were tightened):

    1995      74        2008   1,360        2018   1,968
    2002      38        2010   1,873        2022   2,161
    2006   1,048        2015   1,911        2024   3,476

So Crossref is usable from ~2007 and effectively empty before it. Two further
edges matter as much as the paper counts:

  references    ~99% of covered papers carry them. Layer A' is in good shape
                wherever Layer A exists.
  affiliations  a cliff at 2022 — 7-46 papers/year before, 98% after. T4 and H3
                need affiliation histories and cannot be built from Crossref
                outside that window; T1/T2 use co-authorship and survive.

This bounds a Crossref-only study to the modern era, which costs H5a/H5b their
forty-year series, though H5d (the technique-age lifecycle design) survives.
See docs/08-phase0-findings.md.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator

from .budget import DailyBudget
from .config import require
from .http import ApiClient

log = logging.getLogger(__name__)

BASE = "https://api.crossref.org"

# Crossref member id for the Institute of Electrical and Electronics Engineers,
# confirmed against /members/263 (6.18M DOIs).
IEEE_MEMBER_ID = 263

# Crossref imposes no quota, but a runaway loop is still worth catching, and the
# budget guard is where this project keeps that kind of promise.
RUNAWAY_GUARD = 100_000

# Container-title patterns. Require the venue's canonical phrase *contiguously*,
# which is what separates ICRA from the surprisingly large family of conferences
# whose names contain the same words in a different arrangement. Measured on the
# 2006-2025 harvest, a loose "robotics and automation" match admitted 483 papers
# (2.5% of ICRA) from five other venues:
#
#   ICRAI   International Conference on Robotics and Automation *in Industry*
#   RAHA    ... Robotics and Automation *for Humanitarian Applications*
#   ICRAE   ... Robotics and Automation *Engineering*
#   CCRA    IEEE *Colombian Conference* on Robotics and Automation
#   IMRA    ... Intelligent Manufacturing, Robotics and Automation
#   AIMERA  ... Advanced Information, Mechanical Engineering, Robotics and Automation
#
# The last three break the phrase and are excluded by requiring it contiguously.
# The first three extend it, so the rule is that **nothing but punctuation, a
# parenthesised acronym, or end-of-string may follow** — stated positively rather
# than as a blocklist of qualifier words, since the blocklist approach missed
# ICRAE ("... and Automation Engineering") on the first attempt and would keep
# missing whichever qualifier appears next.
# Genuine containers vary only in year prefix, the "IEEE"/"IEEE/RSJ" tags, a
# trailing "(ICRA)"/"(IROS)", and 2006's "Proceedings ... , 2006. ICRA 2006."
_TAIL = r"(?=\s*(?:\(|,|\.|;|$))"

VENUE_PATTERNS = {
    "IROS": re.compile(
        r"international (?:conference|workshop) on intelligent robots and systems"
        + _TAIL,
        re.I,
    ),
    "ICRA": re.compile(
        r"international conference on robotics and automation" + _TAIL,
        re.I,
    ),
}

# Journals and magazines carrying the same words. The conference patterns above
# already exclude the confusable *conferences*, so this only has to catch serials.
VENUE_EXCLUSIONS = re.compile(
    r"letters|transactions|magazine|journal",
    re.I,
)


def classify_container(container_title: str | None) -> str | None:
    """Return 'ICRA', 'IROS', or None for a Crossref container-title string."""
    if not container_title:
        return None
    if VENUE_EXCLUSIONS.search(container_title):
        return None
    for venue, pattern in VENUE_PATTERNS.items():
        if pattern.search(container_title):
            return venue
    return None


def work_year(work: dict) -> int | None:
    """Publication year, preferring the conference date over the deposit date.

    `created` is when IEEE deposited the record, which for backfilled proceedings
    can be years after the conference — a 2002 IROS paper created in 2003. Using
    it would smear the corpus across the wrong years, so it is the last resort.
    """
    for key in ("published", "published-print", "issued", "published-online"):
        parts = (work.get(key) or {}).get("date-parts") or []
        if parts and parts[0] and parts[0][0]:
            return int(parts[0][0])
    return None


class Crossref:
    def __init__(self, *, dry_run: bool = False, daily_limit: int = RUNAWAY_GUARD):
        self.client = ApiClient(
            "crossref",
            DailyBudget("crossref", daily_limit),
            min_interval=0.05,
            dry_run=dry_run,
        )
        # Crossref's polite pool is identified by a contact address, not a key.
        # Reusing OPENALEX_MAILTO rather than adding a near-duplicate variable.
        self.mailto = require("OPENALEX_MAILTO")

    def _get(self, path: str, params: dict):
        params = dict(params)
        params.setdefault("mailto", self.mailto)
        return self.client.get(f"{BASE}/{path}", params)

    # Relevance-ordered queries used to reach each venue. Two separate queries
    # rather than one combined string: a diluted query ranks both venues worse,
    # which on a relevance-ordered route directly costs recall.
    VENUE_QUERIES = {
        "ICRA": "IEEE International Conference on Robotics and Automation",
        "IROS": "IEEE RSJ International Conference on Intelligent Robots and Systems",
    }

    def iter_venue_year(
        self, venue: str, year: int, *, rows: int = 1000, max_offset: int = 10_000
    ) -> Iterator[dict]:
        """Yield ICRA or IROS papers for one year, relevance-ordered.

        Offset paging, not cursor paging: a fuzzy query is only relevance-ordered
        without a cursor (see module docstring, trap 3). Crossref caps `offset` at
        10,000, which is comfortably above either venue's annual output (~1,200
        papers) but is the reason this cannot be used for an open-ended sweep.

        Stops at the first page yielding no venue match, since relevance ordering
        means matches are front-loaded and the tail is other conferences.
        """
        query = self.VENUE_QUERIES[venue]
        for offset in range(0, max_offset, rows):
            data = self._get("works", {
                "query.container-title": query,
                "filter": (
                    f"type:proceedings-article,"
                    f"from-pub-date:{year}-01-01,until-pub-date:{year}-12-31"
                ),
                "rows": rows,
                "offset": offset,
            })
            if data is None:  # dry run
                return
            items = (data.get("message") or {}).get("items") or []
            if not items:
                return
            matched = 0
            for work in items:
                container = (work.get("container-title") or [None])[0]
                if classify_container(container) == venue:
                    matched += 1
                    yield work
            if matched == 0:
                return

    def survey_year(self, year: int, **kwargs) -> dict:
        """Per-venue counts plus the field availability Phase 1 depends on."""
        out: dict = {}
        for venue in self.VENUE_QUERIES:
            n = refs = affil = 0
            for work in self.iter_venue_year(venue, year, **kwargs):
                n += 1
                if work.get("references-count"):
                    refs += 1
                if any(a.get("affiliation") for a in (work.get("author") or [])):
                    affil += 1
            out[venue] = {"papers": n, "with_references": refs, "with_affiliation": affil}
        return out
