#!/usr/bin/env python3
"""Spend ~8 IEEE calls to determine the entire harvest design.

Answers, per docs/06-roadmap.md Phase 0:
  - Is the key active? ("waiting" status keys return an auth error.)
  - What is the real max_records ceiling, and is deep start_record paging allowed?
  - Which fields come back: abstract, affiliations, index terms, multimedia flags?
  - How far back does abstract coverage actually reach?

Run with --dry-run first to see exactly which calls would be made.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.budget import BudgetExceeded  # noqa: E402
from tacit.ieee import VENUES, IEEEXplore  # noqa: E402

# Spread across four decades: the fields present in a 2023 record tell us nothing
# about a 1986 one, and the early years are exactly where the corpus argument is
# at risk.
PROBE_YEARS = [1986, 1995, 2005, 2015, 2023]


def summarize(article: dict) -> dict:
    """Report field presence, not content — this goes in the log, not the corpus."""
    return {
        "has_abstract": bool(article.get("abstract")),
        "abstract_chars": len(article.get("abstract") or ""),
        "n_authors": len((article.get("authors") or {}).get("authors") or []),
        "has_affiliation": any(
            a.get("affiliation")
            for a in ((article.get("authors") or {}).get("authors") or [])
        ),
        "index_terms": sorted((article.get("index_terms") or {}).keys()),
        "keys": sorted(article.keys()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="data/ieee_probe.json")
    args = ap.parse_args()

    api = IEEEXplore(dry_run=args.dry_run)
    findings: dict = {"years": {}, "errors": []}

    if not args.dry_run:
        print(f"IEEE budget remaining today: {api.remaining_today}/200")

    for year in PROBE_YEARS:
        try:
            data = api.search(
                publication_title=VENUES["ICRA"],
                publication_year=str(year),
                max_records=2,
            )
        except BudgetExceeded as exc:
            findings["errors"].append(str(exc))
            break
        except Exception as exc:  # noqa: BLE001 - probe reports, does not crash
            findings["errors"].append(f"{year}: {type(exc).__name__}: {exc}")
            continue

        if data is None:
            continue
        articles = data.get("articles") or []
        findings["years"][year] = {
            "total_records": data.get("total_records"),
            "n_returned": len(articles),
            "sample": summarize(articles[0]) if articles else None,
        }
        print(f"  {year}: total_records={data.get('total_records')} "
              f"returned={len(articles)}")

    # Ceiling test: ask for more than the assumed maximum and see what comes back.
    try:
        data = api.search(
            publication_title=VENUES["ICRA"],
            publication_year="2015",
            max_records=200,
            start_record=1,
        )
        if data is not None:
            findings["max_records_200_returned"] = len(data.get("articles") or [])
    except Exception as exc:  # noqa: BLE001
        findings["errors"].append(f"ceiling test: {type(exc).__name__}: {exc}")

    if args.dry_run:
        print(f"\nDRY RUN — {len(api.client.planned)} calls would be made:")
        for url in api.client.planned:
            print("  ", url)
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nWrote {out}. Budget remaining: {api.remaining_today}/200")
    if findings["errors"]:
        print("Errors:")
        for err in findings["errors"]:
            print("  ", err)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
