#!/usr/bin/env python3
"""Per-year deflators for the corpus — the denominators every indicator needs.

docs/09-scale-invariance.md sets the rule: no indicator may rest on a raw count.
This script computes what to divide by. It is the study's price index.

Emits, per year: papers, references per paper (mean and median), authors per
paper, distinct authors, and co-authorship graph density both unweighted and
fractionally weighted.

The fractional weighting is not decoration. An n-author paper contributes
n(n-1)/2 co-authorship edges, so a single 279-author consortium paper contributes
~39,000 — more than doubling the 2024 graph by itself. T1 and T2 test lineage on
this graph, so an unweighted version lets one paper redefine "lineage-connected"
for a whole year. Each edge is therefore weighted 1/(n-1): every author of a
paper contributes one unit of collaboration regardless of team size.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import statistics
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def quantile(sorted_vals: list[int], q: float) -> float:
    """Nearest-rank quantile. Reports the shape of a distribution rather than a
    count on one side of an arbitrary line (docs/10 §3)."""
    if not sorted_vals:
        return 0.0
    n = len(sorted_vals)
    # Standard nearest-rank: ceil(q*n) - 1. Using round() on q*(n-1) instead
    # lands on banker's rounding at exact midpoints and shifts the median by one.
    idx = min(n - 1, max(0, math.ceil(q * n) - 1))
    return sorted_vals[idx]


def year_stats(papers: list[dict]) -> dict:
    n = len(papers)
    refs = [p["n_references"] for p in papers]
    # A co-authorship graph counts each *pair* once, however often two people
    # publish together in a year; counting repeat collaborations instead would
    # conflate "how many people you work with" with "how much you publish", and
    # publication volume is itself one of the inflating quantities.
    pairs: set[tuple[str, str]] = set()
    weighted: collections.Counter = collections.Counter()
    authors_seen: set[str] = set()

    for p in papers:
        names = sorted({a["name"] for a in p["authors"] if a.get("name")})
        authors_seen.update(names)
        if len(names) < 2:
            continue
        # One unit of collaboration per author, however large the team.
        w = 1.0 / (len(names) - 1)
        for u, v in itertools.combinations(names, 2):
            pairs.add((u, v))
            weighted[u] += w
            weighted[v] += w

    edges = len(pairs)
    n_authors = len({a for pair in pairs for a in pair})
    # Author-count *distribution*, not a count above a threshold: 20 authors is
    # not different in kind from 19, and the shape is what matters for the
    # fractional weighting in docs/09 §2.
    author_counts = sorted(p["n_authors"] for p in papers)
    return {
        "papers": n,
        "refs_per_paper_mean": round(sum(refs) / n, 2) if n else 0,
        "refs_per_paper_median": statistics.median(refs) if refs else 0,
        "authors_per_paper": round(sum(p["n_authors"] for p in papers) / n, 2) if n else 0,
        "distinct_authors": n_authors,
        "coauthor_edges": edges,
        "mean_degree": round(2 * edges / n_authors, 2) if n_authors else 0,
        "mean_degree_fractional": (
            round(sum(weighted.values()) / n_authors, 2) if n_authors else 0
        ),
        "with_references": sum(1 for p in papers if p["n_references"]),
        "with_affiliation": sum(1 for p in papers if p["n_affiliations"]),
        "authors_median": quantile(author_counts, 0.50),
        "authors_p90": quantile(author_counts, 0.90),
        "authors_p99": quantile(author_counts, 0.99),
        "authors_max": author_counts[-1] if author_counts else 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/crossref_works.jsonl")
    ap.add_argument("--out", default="data/corpus_baselines.json")
    args = ap.parse_args()

    papers = load(Path(args.corpus))
    by_year: dict[int, list[dict]] = collections.defaultdict(list)
    for p in papers:
        by_year[p["year"]].append(p)

    stats = {y: year_stats(by_year[y]) for y in sorted(by_year)}

    header = (f"{'year':<6}{'papers':>7}{'refs/pp':>9}{'auth/pp':>9}"
              f"{'authors':>9}{'degree':>8}{'frac deg':>10}{'a-p99':>7}{'a-max':>7}")
    print(header)
    print("-" * len(header))
    for year, s in stats.items():
        print(f"{year:<6}{s['papers']:>7,}{s['refs_per_paper_mean']:>9.1f}"
              f"{s['authors_per_paper']:>9.2f}{s['distinct_authors']:>9,}"
              f"{s['mean_degree']:>8.2f}{s['mean_degree_fractional']:>10.2f}"
              f"{s['authors_p99']:>7.0f}{s['authors_max']:>7}")

    first, last = min(stats), max(stats)
    print(f"\nInflation {first}->{last}:")
    for key, label in [("papers", "papers/year"),
                       ("refs_per_paper_mean", "references/paper"),
                       ("authors_per_paper", "authors/paper"),
                       ("mean_degree", "co-author degree (raw)"),
                       ("mean_degree_fractional", "co-author degree (fractional)")]:
        a, b = stats[first][key], stats[last][key]
        if a:
            print(f"  {label:<32} {a:>8.2f} -> {b:>8.2f}   x{b / a:.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, default=str))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
