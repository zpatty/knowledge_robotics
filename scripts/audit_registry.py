#!/usr/bin/env python3
"""Emit a hand-checkable sample of technique matches, plus precision diagnostics.

Title-substring matching has checkable precision and unmeasurable recall
(docs/11). This script makes the checkable half easy: for every technique it
prints a random sample of matched titles and which alias fired, so a domain
reader can mark false positives without reading code or rerunning anything.

It also flags two patterns worth looking at before trusting a technique's counts:

  * an alias that fires on a large share of the whole corpus, which usually means
    the term is generic rather than a technique name;
  * an alias contributing matches that no other alias for that technique also
    matches, which is where a single loose string is carrying a technique on its
    own and a false positive would be invisible.

Neither flag is a filter. Nothing is dropped — the output is for a human to read.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tacit.registry import load_registry, normalise_title  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/crossref_works.jsonl")
    ap.add_argument("--out", default="data/registry_audit.md")
    ap.add_argument("--sample", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    papers = [json.loads(line) for line in Path(args.corpus).open() if line.strip()]
    registry = load_registry()
    rng = random.Random(args.seed)

    # technique -> alias -> [titles]
    hits: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    matched_papers: dict[str, set[int]] = defaultdict(set)

    for idx, paper in enumerate(papers):
        padded = normalise_title(paper.get("title"))
        for tech in registry:
            if any(bad in padded for bad in tech.exclude):
                continue
            fired = [a for a in tech.aliases if a in padded]
            if not fired:
                continue
            matched_papers[tech.id].add(idx)
            for alias in fired:
                hits[tech.id][alias].append(paper.get("title") or "")

    total = len(papers)
    lines: list[str] = [
        "# Technique registry audit",
        "",
        f"Corpus: {total:,} papers. Sample of {args.sample} titles per technique, "
        f"seed {args.seed}.",
        "",
        "Mark false positives and edit `registry/techniques.json` — `aliases` to "
        "widen or narrow, `exclude` to remove a known bad pattern. Nothing here "
        "filters anything; it is for reading.",
        "",
    ]

    for tech in registry:
        n = len(matched_papers[tech.id])
        share = n / total
        lines.append(f"## {tech.id} — {tech.name}")
        lines.append("")
        lines.append(f"axis: `{tech.axis}` · matched **{n:,}** papers ({share:.1%})")
        lines.append("")

        alias_rows = sorted(hits[tech.id].items(), key=lambda kv: -len(kv[1]))
        lines.append("| alias | matches | share of corpus | sole-carrier |")
        lines.append("|---|---|---|---|")
        for alias, titles in alias_rows:
            # Does this alias carry papers no sibling alias also matches?
            others = {t for a, ts in alias_rows if a != alias for t in ts}
            sole = len([t for t in titles if t not in others])
            flag = ""
            if len(titles) / total > 0.02:
                flag = " ⚠ generic?"
            lines.append(
                f"| `{alias}` | {len(titles):,} | {len(titles) / total:.2%} | "
                f"{sole:,}{flag} |"
            )
        lines.append("")

        sample_titles = sorted({t for ts in hits[tech.id].values() for t in ts})
        if sample_titles:
            picked = rng.sample(sample_titles, min(args.sample, len(sample_titles)))
            lines.append("Sample matches:")
            lines.append("")
            for title in picked:
                lines.append(f"- {title[:150]}")
            lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))

    flagged = [
        (tech.id, alias, len(titles))
        for tech in registry
        for alias, titles in hits[tech.id].items()
        if len(titles) / total > 0.02
    ]
    print(f"Wrote {out}")
    print(f"techniques: {len(registry)}  "
          f"papers matched: {len(set().union(*matched_papers.values())):,}/{total:,}")
    if flagged:
        print("\nAliases matching >2% of the corpus — check these first:")
        for tid, alias, n in sorted(flagged, key=lambda x: -x[2]):
            print(f"  {tid:<24}{alias!r:<32}{n:,} ({n / total:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
