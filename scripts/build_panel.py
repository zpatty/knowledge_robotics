#!/usr/bin/env python3
"""Build the technique x year panel — the study's main analysis table.

docs/04 section 2 calls for a technique x year table carrying every computable
indicator. This builds the part reachable from Layers A/A' (docs/11): adoption,
B2 reference reach and age, and the author-cohort quantities the transfer channel
needs.

Deliberately simple, per docs/11: one pass over the corpus, plain dicts, CSV out.
No modelling, no interpretation, no derived scores beyond the ratios each
indicator is defined as.

Every count is emitted alongside its deflator (docs/09): `papers` never travels
without `share_of_year`, and `n_authors` never without `share_of_year_authors`.
Raw counts are kept in the file because dropping them would stop anyone checking
the ratios — the rule is that no *indicator* rests on a raw count, not that counts
may not be recorded.

Outputs
    data/panel.csv              technique x year
    data/panel_corpus.csv       corpus totals per year (the denominators)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.refs import ReferenceClassifier, summarise  # noqa: E402
from tacit.registry import load_registry, match_paper  # noqa: E402


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/crossref_works.jsonl")
    ap.add_argument("--out", default="data/panel.csv")
    ap.add_argument("--corpus-out", default="data/panel_corpus.csv")
    args = ap.parse_args()

    papers = [json.loads(line) for line in Path(args.corpus).open() if line.strip()]
    registry = load_registry()
    classifier = ReferenceClassifier()
    axis = {t.id: t.axis for t in registry}
    names = {t.id: t.name for t in registry}

    # Per-year corpus totals: the denominators everything else is divided by.
    corpus_papers: dict[int, int] = defaultdict(int)
    corpus_authors: dict[int, set[str]] = defaultdict(set)
    corpus_refs: dict[int, list[int]] = defaultdict(list)

    cells: dict[tuple[str, int], dict] = defaultdict(
        lambda: {"papers": 0, "authors": set(), "reach": [], "classified": [],
                 "age_mean": [], "age_median": [], "refs": [], "fields": defaultdict(int)}
    )
    # First year each author appears anywhere, so "entrants" means new to the
    # corpus rather than new to the technique.
    first_seen: dict[str, int] = {}

    for paper in papers:
        year = paper["year"]
        author_names = [a["name"] for a in paper["authors"] if a.get("name")]
        corpus_papers[year] += 1
        corpus_authors[year].update(author_names)
        corpus_refs[year].append(paper["n_references"])
        for name in author_names:
            if name not in first_seen or year < first_seen[name]:
                first_seen[name] = year

        technique_ids = match_paper(paper, registry)
        if not technique_ids:
            continue
        b2 = summarise(paper["references"], year, classifier)
        for tid in technique_ids:
            cell = cells[(tid, year)]
            cell["papers"] += 1
            cell["authors"].update(author_names)
            cell["reach"].append(b2["reach"])
            cell["classified"].append(b2["classified_share"])
            cell["age_mean"].append(b2["age_mean"])
            cell["age_median"].append(b2["age_median"])
            cell["refs"].append(b2["n_references"])
            for field, count in b2["external_fields"].items():
                cell["fields"][field] += count

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "technique", "technique_name", "axis", "year",
            "papers", "share_of_year",
            "n_authors", "share_of_year_authors", "entrant_share",
            "b2_reach", "b2_classified_share", "ref_age_mean", "ref_age_median",
            "refs_per_paper", "top_external_field",
        ])
        for (tid, year) in sorted(cells, key=lambda k: (k[0], k[1])):
            cell = cells[(tid, year)]
            year_papers = corpus_papers[year] or 1
            year_authors = len(corpus_authors[year]) or 1
            authors = cell["authors"]
            entrants = sum(1 for a in authors if first_seen.get(a) == year)
            top_field = max(cell["fields"], key=cell["fields"].get) if cell["fields"] else ""
            writer.writerow([
                tid, names[tid], axis[tid], year,
                cell["papers"], round(cell["papers"] / year_papers, 6),
                len(authors), round(len(authors) / year_authors, 6),
                round(entrants / len(authors), 4) if authors else "",
                _r(mean(cell["reach"]), 4), _r(mean(cell["classified"]), 4),
                _r(mean(cell["age_mean"]), 2), _r(mean(cell["age_median"]), 2),
                _r(mean(cell["refs"]), 2), top_field,
            ])

    corpus_out = Path(args.corpus_out)
    with corpus_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year", "papers", "distinct_authors", "refs_per_paper"])
        for year in sorted(corpus_papers):
            writer.writerow([
                year, corpus_papers[year], len(corpus_authors[year]),
                round(mean(corpus_refs[year]) or 0, 2),
            ])

    print(f"Wrote {out} ({len(cells):,} technique-year cells)")
    print(f"Wrote {corpus_out} ({len(corpus_papers)} years)")
    return 0


def _r(value, digits):
    return "" if value is None else round(value, digits)


if __name__ == "__main__":
    raise SystemExit(main())
