#!/usr/bin/env python3
"""The Phase 0 gate: abstract and metadata coverage by year.

docs/03-corpus.md argues that complete abstracts make the abstract-level
indicators corpus-wide and unbiased, which is what contains the openness-selection
threat. That argument stands or falls on this table, so it is computed before
anything else is built on top of it.

Prints, per venue-year: works, abstract coverage, reference coverage, affiliation
coverage, and how many works would need an IEEE call to fill the abstract gap
(which is what the scarce IEEE budget should be spent on).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--works", default="data/openalex_works.jsonl")
    ap.add_argument("--csv", default="data/coverage_by_year.csv")
    args = ap.parse_args()

    path = Path(args.works)
    if not path.exists():
        print(f"{path} not found — run harvest_openalex.py first", file=sys.stderr)
        return 2

    rows = defaultdict(lambda: defaultdict(int))
    for line in path.open():
        w = json.loads(line)
        year = w.get("publication_year")
        src = ((w.get("primary_location") or {}).get("source") or {})
        venue = (src.get("display_name") or "?")[:40]
        key = (venue, year)
        rows[key]["works"] += 1
        if w.get("abstract_text"):
            rows[key]["abstract"] += 1
        if w.get("referenced_works"):
            rows[key]["refs"] += 1
        auths = w.get("authorships") or []
        if any(a.get("institutions") for a in auths):
            rows[key]["affil"] += 1

    out = Path(args.csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "venue,year,works,abstract_pct,refs_pct,affil_pct,ieee_calls_to_fill"
    lines = [header]
    print(f"{'venue':<40} {'year':>5} {'works':>6} {'abs%':>6} {'ref%':>6} {'aff%':>6}")
    for (venue, year), c in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        n = c["works"] or 1
        a, r, f = 100 * c["abstract"] / n, 100 * c["refs"] / n, 100 * c["affil"] / n
        # IEEE returns up to ~200 records/call; gaps are what the budget is for.
        gap_calls = -(-(c["works"] - c["abstract"]) // 200)
        print(f"{venue:<40} {year or 0:>5} {c['works']:>6} {a:>5.1f}% {r:>5.1f}% {f:>5.1f}%")
        lines.append(f'"{venue}",{year},{c["works"]},{a:.1f},{r:.1f},{f:.1f},{gap_calls}')
    out.write_text("\n".join(lines) + "\n")

    total_gap = sum(-(-(c["works"] - c["abstract"]) // 200) for c in rows.values())
    print(f"\nWrote {out}")
    print(f"Filling every abstract gap from IEEE would cost ~{total_gap} calls "
          f"(~{-(-total_gap // 200)} days at 200/day).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
