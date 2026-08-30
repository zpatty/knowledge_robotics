#!/usr/bin/env python3
"""Fetch papers by DOI — the route into the era Crossref cannot enumerate.

docs/08 F4 established the asymmetry: pre-2007 ICRA/IROS records are *in*
Crossref and resolve perfectly by DOI, but are unreachable through its search
index. So anything that supplies a DOI list converts that era from blocked to a
straightforward fetch.

The list used here comes from the corpus's own reference edges: 6,210 distinct
pre-2006 ICRA/IROS DOIs cited by the 2006-2025 papers, spanning 1984-2005
continuously (scripts/build_pre2006_list.py).

**This is a cited subset, not a census, and the distinction is load-bearing.**
Only papers someone later cited are reachable this way, so it over-represents
influential and enduring work, and the bias worsens going back — roughly 15 of
1984's ~200 papers. Any corpus-level or per-year-rate claim computed on it is
wrong. It is defensible for the transfer channel, where lineage can only be
traced through cited work anyway, and for technique-level comparisons within the
same selection. Every record is written with `selection: "cited-subset"` so the
distinction survives into the panel rather than living only in this docstring.

A DBLP or IEEE venue index would replace this with a real census; until one
arrives this is the best available and it costs nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.crossref import Crossref, classify_container, work_year  # noqa: E402
from tacit.http import ApiError  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harvest_crossref import normalise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dois", default="data/pre2006_doi_candidates.json")
    ap.add_argument("--out", default="data/crossref_pre2006.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    by_year = json.loads(Path(args.dois).read_text())
    wanted = [d for year in sorted(by_year) for d in by_year[year]]

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
                seen.add(doi.lower())
        print(f"resuming: {len(seen):,} already fetched")

    todo = [d for d in wanted if d.lower() not in seen]
    print(f"{len(wanted):,} DOIs on the list, {len(todo):,} to fetch")
    if args.dry_run:
        for d in todo[:10]:
            print("  ", d)
        return 0

    api = Crossref()
    written = failed = wrong_venue = undated = 0
    t0 = time.time()
    with out.open("a") as fh:
        for i, doi in enumerate(todo, 1):
            try:
                data = api._get(f"works/{doi}", {})
            except ApiError:
                failed += 1
                continue
            except Exception:  # noqa: BLE001 - one bad record must not end the run
                failed += 1
                continue
            work = (data or {}).get("message") or {}
            venue = classify_container((work.get("container-title") or [None])[0])
            if not venue:
                wrong_venue += 1
                continue
            record = normalise(work, venue, work_year(work) or 0)
            if not record["year"]:
                undated += 1
            record["selection"] = "cited-subset"
            fh.write(json.dumps(record) + "\n")
            written += 1
            if i % 250 == 0:
                fh.flush()
                rate = i / max(time.time() - t0, 1e-9)
                print(f"  {i:,}/{len(todo):,}  written {written:,}  "
                      f"failed {failed}  off-venue {wrong_venue}  "
                      f"({rate:.1f}/s)", flush=True)

    print(f"\nWrote {written:,} papers to {out}")
    print(f"  failed {failed:,} · off-venue {wrong_venue:,} · "
          f"undated {undated:,} · {time.time() - t0:.0f}s")
    print(f"  cache hits={api.client.hits} misses={api.client.misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
