#!/usr/bin/env python3
"""Extract pre-2006 ICRA/IROS DOIs from the corpus's own reference lists.

The 2006-2025 papers cite the era we cannot enumerate. IEEE conference DOIs
carry the venue and year in the suffix (10.1109/ROBOT.1998.677043), so the list
falls out of the reference edges with a regex and no API calls at all.

Yields a cited subset, never a census - see harvest_dois.py for what that
forbids.
"""
from __future__ import annotations

import argparse, collections, json, re
from pathlib import Path

# robot = ICRA pre-2007, iros/irds = IROS, icra = later ICRA form.
STEM = re.compile(r'^10\.1109/(robot|iros|irds|icra)[.\d]*\.(\d{4})\.', re.I)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/crossref_works.jsonl")
    ap.add_argument("--out", default="data/pre2006_doi_candidates.json")
    ap.add_argument("--until", type=int, default=2005)
    ap.add_argument("--since", type=int, default=1984)
    args = ap.parse_args()

    found: dict[int, set[str]] = collections.defaultdict(set)
    for line in Path(args.corpus).open():
        if not line.strip():
            continue
        for ref in json.loads(line)["references"]:
            m = STEM.match(ref.get("doi") or "")
            if m and args.since <= int(m.group(2)) <= args.until:
                found[int(m.group(2))].add(ref["doi"].lower())

    Path(args.out).write_text(json.dumps(
        {str(y): sorted(v) for y, v in sorted(found.items())}, indent=1))
    total = sum(len(v) for v in found.values())
    print(f"{total:,} distinct DOIs, {min(found)}-{max(found)}")
    for y in sorted(found):
        print(f"  {y}  {len(found[y]):>5,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
