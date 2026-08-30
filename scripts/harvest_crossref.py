#!/usr/bin/env python3
"""Harvest the reachable ICRA/IROS corpus from Crossref into a JSONL panel.

Layers A and A′ for ~2007 onward — the window `survey_crossref.py` established.
That is enough for the whole Phase 1 study (docs/03 §3.2): technique registry,
actor graph, T1/T2, S1/S2/S4 where artifact links exist, and B2.

Writes one JSON object per paper to `--out`, keeping only the fields the study
needs. Records are normalised at write time so the analysis layer never has to
know about Crossref's shape:

    venue, year, doi, title, container, authors[], affiliations[],
    references[] (doi + unstructured), n_references, n_refs_with_doi,
    subject[], isbn, page, publisher, license

Resumable: an existing --out is read first and its DOIs skipped, so an
interrupted run continues rather than restarting. Crossref is unmetered, so the
cost of a re-run is time, and the response cache makes a repeat nearly free.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.crossref import Crossref, work_year  # noqa: E402


def parse_years(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def normalise(work: dict, venue: str, year: int) -> dict:
    authors = []
    affiliations = []
    for a in work.get("author") or []:
        name = " ".join(p for p in (a.get("given"), a.get("family")) if p).strip()
        authors.append({
            "name": name or a.get("name"),
            "family": a.get("family"),
            "orcid": a.get("ORCID"),
            "sequence": a.get("sequence"),
        })
        for aff in a.get("affiliation") or []:
            if aff.get("name"):
                affiliations.append(aff["name"])

    references = []
    for r in work.get("reference") or []:
        references.append({
            "doi": r.get("DOI"),
            "year": r.get("year"),
            "title": r.get("article-title"),
            "venue": r.get("journal-title"),
            "unstructured": r.get("unstructured"),
        })

    return {
        "venue": venue,
        "year": year,
        "doi": work.get("DOI"),
        "title": (work.get("title") or [None])[0],
        "container": (work.get("container-title") or [None])[0],
        "type": work.get("type"),
        "authors": authors,
        "n_authors": len(authors),
        "affiliations": affiliations,
        "n_affiliations": len(affiliations),
        "references": references,
        "n_references": len(references),
        "n_refs_with_doi": sum(1 for r in references if r["doi"]),
        "subject": work.get("subject") or [],
        "isbn": (work.get("ISBN") or [None])[0],
        "page": work.get("page"),
        "publisher": work.get("publisher"),
        "license": [l.get("URL") for l in (work.get("license") or [])],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2007-2025")
    ap.add_argument("--out", default="data/crossref_works.jsonl")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = Crossref(dry_run=args.dry_run)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if out.exists():
        for line in out.open():
            try:
                doi = json.loads(line).get("doi")
            except json.JSONDecodeError:
                continue
            if doi:
                seen.add(doi)
        print(f"Resuming: {len(seen):,} papers already in {out}")

    totals = {"ICRA": 0, "IROS": 0}
    refs_total = affil_total = written = 0
    t0 = time.time()

    header = f"{'year':<6}{'ICRA':>7}{'IROS':>7}{'new':>8}{'w/refs':>8}{'w/affil':>9}"
    print(header)
    print("-" * len(header))

    with out.open("a") as fh:
        for year in parse_years(args.years):
            per_year = {"ICRA": 0, "IROS": 0}
            new = yrefs = yaffil = 0
            for venue in ("ICRA", "IROS"):
                for work in api.iter_venue_year(venue, year, rows=args.rows):
                    per_year[venue] += 1
                    doi = work.get("DOI")
                    if not doi or doi in seen:
                        continue
                    seen.add(doi)
                    # Trust the record's own date over the query's year filter.
                    record = normalise(work, venue, work_year(work) or year)
                    fh.write(json.dumps(record) + "\n")
                    new += 1
                    written += 1
                    if record["n_references"]:
                        yrefs += 1
                    if record["n_affiliations"]:
                        yaffil += 1
            fh.flush()
            totals["ICRA"] += per_year["ICRA"]
            totals["IROS"] += per_year["IROS"]
            refs_total += yrefs
            affil_total += yaffil
            print(f"{year:<6}{per_year['ICRA']:>7,}{per_year['IROS']:>7,}"
                  f"{new:>8,}{yrefs:>8,}{yaffil:>9,}")

    if args.dry_run:
        print(f"\nDRY RUN — {len(api.client.planned)} calls would be made")
        return 0

    print(f"\nWrote {written:,} new papers to {out} in {time.time() - t0:.0f}s")
    print(f"  ICRA {totals['ICRA']:,} | IROS {totals['IROS']:,} | "
          f"with refs {refs_total:,} | with affiliations {affil_total:,}")
    print(f"  cache hits={api.client.hits} misses={api.client.misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
