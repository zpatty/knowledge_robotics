#!/usr/bin/env python3
"""Bulk-harvest ICRA/IROS from OpenAlex — metadata, references, abstracts.

Two stages, deliberately separated so the expensive one is never run on a guess:

  --stage sources : find and print candidate venue source records. A long-running
                    conference series is frequently split across several OpenAlex
                    source records (renamed proceedings, per-year records), and
                    picking one at random silently truncates the corpus. Inspect
                    the output and record the chosen IDs before harvesting.
  --stage works   : cursor-page every work for the given source IDs.

Then scripts/coverage_report.py answers the Phase 0 gating question.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.openalex import OpenAlex, decode_abstract  # noqa: E402


def stage_sources(api: OpenAlex) -> None:
    for query in ("International Conference on Robotics and Automation",
                  "Intelligent Robots and Systems"):
        print(f"\n=== {query} ===")
        for src in api.find_sources(query):
            print(f"{src.get('id')}  works={src.get('works_count'):>7}  "
                  f"type={src.get('type'):<12} {src.get('display_name')}")
    print("\nRecord the chosen source IDs, then re-run with --stage works --sources ...")


def stage_works(api: OpenAlex, source_ids: list[str], out: Path) -> None:
    ids = "|".join(source_ids)
    filters = f"primary_location.source.id:{ids}"

    total = api.count_works(filters)
    print(f"OpenAlex reports {total:,} works for these sources.")
    print(f"At per-page=100 that is ~{-(-total // 100):,} calls.")

    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as fh:
        for work in api.iter_works(filters):
            work["abstract_text"] = decode_abstract(work.pop("abstract_inverted_index", None))
            fh.write(json.dumps(work) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"  {written:,}/{total:,}")
    print(f"Wrote {written:,} works to {out}")
    print(f"cache: hits={api.client.hits} misses={api.client.misses}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sources", "works"], required=True)
    ap.add_argument("--sources", nargs="*", default=[],
                    help="OpenAlex source IDs (e.g. S4306419644)")
    ap.add_argument("--out", default="data/openalex_works.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = OpenAlex(dry_run=args.dry_run)
    if args.stage == "sources":
        stage_sources(api)
    else:
        if not args.sources:
            print("--sources is required for --stage works", file=sys.stderr)
            return 2
        stage_works(api, args.sources, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
