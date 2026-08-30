#!/usr/bin/env python3
"""Per-year ICRA/IROS coverage in Crossref — how much corpus is reachable today.

Answers what docs/08-phase0-findings.md left open: OpenAlex resolves IROS to
three years out of thirty-five, so can Crossref supply the rest? Short answer,
from this script: only back to about 2008.

Crossref is unmetered, so this costs time rather than budget. Each venue-year is
at most ten calls and usually two.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.crossref import Crossref  # noqa: E402


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="1984-2025")
    ap.add_argument("--out", default="data/crossref_survey.json")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = Crossref(dry_run=args.dry_run)
    findings: dict = {"years": {}}

    header = f"{'year':<6}{'ICRA':>7}{'IROS':>7}{'total':>8}{'w/refs':>8}{'w/affil':>9}"
    print(header)
    print("-" * len(header))
    for year in parse_years(args.years):
        survey = api.survey_year(year, rows=args.rows)
        icra, iros = survey["ICRA"], survey["IROS"]
        total = icra["papers"] + iros["papers"]
        refs = icra["with_references"] + iros["with_references"]
        affil = icra["with_affiliation"] + iros["with_affiliation"]
        findings["years"][year] = survey
        print(f"{year:<6}{icra['papers']:>7,}{iros['papers']:>7,}{total:>8,}"
              f"{refs:>8,}{affil:>9,}")

    if args.dry_run:
        print(f"\nDRY RUN — {len(api.client.planned)} calls would be made")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(findings, indent=2))
    print(f"\nWrote {out}   (cache hits={api.client.hits} misses={api.client.misses})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
