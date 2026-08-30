#!/usr/bin/env python3
"""Rolling lineage proximity per technique-year — the continuous T1/T2.

For technique T and year Y: how close are that year's *new* adopters of T to
everyone who worked on T before Y?

No cohort split. docs/10 rules out arbitrary cut points, and an earlier probe
split adopters at 2015 — an arbitrary cut inside a demonstration against
arbitrary cuts. Here each year is its own observation against all prior years, so
the panel carries a time series rather than one before/after contrast, and
nothing is gated.

Reported as observed proximity over its degree-matched null (docs/10 section 2),
which is continuous and scale-free at once. `alpha` is swept rather than fixed.

Resumable: rows are appended and flushed as they are computed, and an existing
output is read first so an interrupted run continues. This matters because a full
sweep is hours — the walk needs epsilon=1e-6 to converge (docs/10), and that is
~15s per null draw.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.lineage import (  # noqa: E402
    build_coauthor_graph, degree_buckets, degrees, proximity_vs_null,
)
from tacit.registry import load_registry, match_paper  # noqa: E402

FIELDS = [
    "technique", "year", "alpha", "n_prior_authors", "n_new_authors",
    "observed", "expected", "ratio", "z", "n_null", "seconds",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/crossref_works.jsonl")
    ap.add_argument("--out", default="data/indicator_lineage.csv")
    ap.add_argument("--alphas", default="0.05,0.15,0.30")
    ap.add_argument("--n-null", type=int, default=5)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--min-authors", type=int, default=3,
                    help="compute budget only: cells with fewer new or prior "
                         "authors are recorded as skipped, never silently dropped")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    papers = [json.loads(line) for line in Path(args.corpus).open() if line.strip()]
    registry = load_registry()

    graph = build_coauthor_graph(papers)
    deg_w = degrees(graph)
    buckets = degree_buckets(graph)
    print(f"graph: {len(graph):,} authors, "
          f"{sum(len(v) for v in graph.values()) // 2:,} edges")

    # technique -> year -> set of author names
    by_tech: dict[str, dict[int, set[str]]] = defaultdict(lambda: defaultdict(set))
    for paper in papers:
        names = {a["name"] for a in paper["authors"] if a.get("name")}
        for tid in match_paper(paper, registry):
            by_tech[tid][paper["year"]] |= names

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int, str]] = set()
    if out.exists():
        with out.open() as fh:
            for row in csv.DictReader(fh):
                done.add((row["technique"], int(row["year"]), row["alpha"]))
        print(f"resuming: {len(done):,} cells already computed")

    new_file = not out.exists()
    with out.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for tid in sorted(by_tech):
            years = sorted(by_tech[tid])
            prior: set[str] = set()
            for year in years:
                current = by_tech[tid][year]
                new_authors = sorted(current - prior)
                prior_authors = sorted(prior)
                for alpha in alphas:
                    key = (tid, year, str(alpha))
                    if key in done:
                        continue
                    if (len(prior_authors) < args.min_authors
                            or len(new_authors) < args.min_authors):
                        writer.writerow({
                            "technique": tid, "year": year, "alpha": alpha,
                            "n_prior_authors": len(prior_authors),
                            "n_new_authors": len(new_authors),
                            "observed": "", "expected": "", "ratio": "",
                            "z": "", "n_null": 0, "seconds": 0,
                        })
                        continue
                    t0 = time.time()
                    res = proximity_vs_null(
                        graph, prior_authors, new_authors,
                        alpha=alpha, n_null=args.n_null, epsilon=args.epsilon,
                        degree=deg_w, degree_buckets=buckets,
                    )
                    writer.writerow({
                        "technique": tid, "year": year, "alpha": alpha,
                        "n_prior_authors": len(prior_authors),
                        "n_new_authors": len(new_authors),
                        "observed": round(res["observed"], 8),
                        "expected": round(res["expected"], 8),
                        "ratio": round(res["ratio"], 4),
                        "z": round(res["z"], 3),
                        "n_null": res.get("n_null", 0),
                        "seconds": round(time.time() - t0, 1),
                    })
                    fh.flush()
                    print(f"  {tid:<24}{year}  a={alpha:<5} ratio={res['ratio']:.2f} "
                          f"z={res['z']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
                prior |= current

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
