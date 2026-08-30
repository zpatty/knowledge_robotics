#!/usr/bin/env python3
"""Verify the push tolerance across real cells before trusting the lineage sweep.

epsilon=1e-6 was chosen from ONE impedance-control cell (docs/10). A tolerance
that is too loose truncates the walk and acts as a distance cutoff, so it has to
be checked on the cells actually being computed, across a spread of cohort sizes.
If 1e-6 and 1e-7 disagree materially, the sweep is truncated and must be rerun.
"""
from __future__ import annotations

import csv, json, random, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tacit.lineage import build_coauthor_graph, degrees, proximity  # noqa: E402
from tacit.registry import load_registry, match_paper  # noqa: E402

papers = [json.loads(l) for l in Path("data/crossref_works.jsonl").open() if l.strip()]
registry = load_registry()
graph = build_coauthor_graph(papers)
deg = degrees(graph)

by_tech: dict[str, dict[int, set]] = defaultdict(lambda: defaultdict(set))
for p in papers:
    names = {a["name"] for a in p["authors"] if a.get("name")}
    for tid in match_paper(p, registry):
        by_tech[tid][p["year"]] |= names

cells = []
for tid, years in by_tech.items():
    prior: set = set()
    for year in sorted(years):
        new = years[year] - prior
        if len(prior) >= 3 and len(new) >= 3:
            cells.append((tid, year, sorted(prior), sorted(new)))
        prior |= years[year]

rng = random.Random(0)
# Spread across cohort sizes rather than sampling uniformly: truncation should
# bite hardest where the prior set is large and the walk has furthest to spread.
cells.sort(key=lambda c: len(c[2]))
picks = [cells[0], cells[len(cells)//4], cells[len(cells)//2],
         cells[3*len(cells)//4], cells[-1]] + rng.sample(cells, 5)

print(f"{'technique':<24}{'year':>5}{'prior':>7}{'new':>6}"
      f"{'eps=1e-5':>11}{'eps=1e-6':>11}{'eps=1e-7':>11}{'6v7 diff':>10}")
worst = 0.0
for tid, year, prior, new in picks:
    vals = {}
    for eps in (1e-5, 1e-6, 1e-7):
        t0 = time.time()
        vals[eps] = proximity(graph, prior, new, alpha=0.15, epsilon=eps, degree=deg)
    a, b = vals[1e-6], vals[1e-7]
    rel = abs(a - b) / b if b else 0.0
    worst = max(worst, rel)
    print(f"{tid:<24}{year:>5}{len(prior):>7}{len(new):>6}"
          f"{vals[1e-5]:>11.6f}{a:>11.6f}{b:>11.6f}{rel:>9.1%}")
print(f"\nworst 1e-6 vs 1e-7 relative difference: {worst:.2%}")
print("VERDICT:", "1e-6 converged" if worst < 0.05 else "1e-6 TRUNCATED - rerun tighter")
