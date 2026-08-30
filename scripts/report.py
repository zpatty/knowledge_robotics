#!/usr/bin/env python3
"""Descriptive report over the panel — distributions, not conclusions.

docs/11 is explicit that this pass produces data. So this prints what is in the
panel and stops: coverage, distributions, and per-technique series. It draws no
inferences, ranks nothing as evidence, and tests no hypothesis. Where a number
looks like a finding it is labelled as descriptive, because the confounds in
docs/10 section 2 (cohort growth) and docs/08 F5 (no ORCID, no early
affiliations) are unresolved and every one of them cuts across these columns.

Reads whatever exists; missing inputs are reported and skipped, never waited on.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def num(row: dict, key: str):
    value = row.get(key, "")
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def block(title: str) -> list[str]:
    return ["", f"## {title}", ""]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel.csv")
    ap.add_argument("--corpus", default="data/panel_corpus.csv")
    ap.add_argument("--lineage", default="data/indicator_lineage.csv")
    ap.add_argument("--out", default="data/report.md")
    args = ap.parse_args()

    panel = read_csv(Path(args.panel))
    corpus = read_csv(Path(args.corpus))
    lineage = read_csv(Path(args.lineage))

    out: list[str] = [
        "# Panel report",
        "",
        "Descriptive output over `data/panel.csv`. **Nothing here is a finding.** "
        "The cohort-growth confound (`docs/10` §2) and the identity limits "
        "(`docs/08` F5 — no ORCID, no affiliations before 2019) are unresolved, "
        "and both cut across every column below.",
    ]

    # ---- corpus ----
    out += block("Corpus")
    if corpus:
        out += ["| year | papers | distinct authors | refs/paper |", "|---|---|---|---|"]
        for row in corpus:
            out.append(f"| {row['year']} | {int(row['papers']):,} | "
                       f"{int(row['distinct_authors']):,} | {row['refs_per_paper']} |")
        total = sum(int(r["papers"]) for r in corpus)
        out += ["", f"Total {total:,} papers across {len(corpus)} years."]
    else:
        out.append("_missing_")

    # ---- technique coverage ----
    out += block("Technique coverage")
    per_tech: dict[str, list[dict]] = defaultdict(list)
    for row in panel:
        per_tech[row["technique"]].append(row)
    out += ["| technique | axis | years | papers | peak share of year |",
            "|---|---|---|---|---|"]
    for tid in sorted(per_tech, key=lambda t: -sum(int(r["papers"]) for r in per_tech[t])):
        rows = per_tech[tid]
        papers = sum(int(r["papers"]) for r in rows)
        peak = max(num(r, "share_of_year") or 0 for r in rows)
        out.append(f"| `{tid}` | {rows[0]['axis']} | {len(rows)} | {papers:,} | {peak:.2%} |")

    # ---- B2 ----
    out += block("B2 — reference reach (descriptive)")
    out.append("Fraction of *classified* references falling outside robotics. "
               "`classified share` is the denominator's coverage; a low value "
               "means the reach figure rests on a small part of the reference "
               "list (`docs/08` F5).")
    out.append("")
    by_axis: dict[str, list[float]] = defaultdict(list)
    for row in panel:
        reach = num(row, "b2_reach")
        if reach is not None:
            by_axis[row["axis"]].append(reach)
    out += ["| axis | mean reach | median | n technique-years |", "|---|---|---|---|"]
    for axis, values in sorted(by_axis.items(), key=lambda kv: -statistics.mean(kv[1])):
        out.append(f"| {axis} | {statistics.mean(values):.3f} | "
                   f"{statistics.median(values):.3f} | {len(values)} |")

    out += ["", "By year, pooled across techniques:", "",
            "| year | mean reach | mean classified share | mean ref age |",
            "|---|---|---|---|"]
    by_year: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in panel:
        for key in ("b2_reach", "b2_classified_share", "ref_age_mean"):
            value = num(row, key)
            if value is not None:
                by_year[row["year"]][key].append(value)
    for year in sorted(by_year):
        cell = by_year[year]
        out.append(
            f"| {year} | {_m(cell['b2_reach'])} | "
            f"{_m(cell['b2_classified_share'])} | {_m(cell['ref_age_mean'])} |"
        )

    # ---- lineage ----
    out += block("Lineage proximity (continuous T1/T2)")
    if lineage:
        computed = [r for r in lineage if r.get("ratio") not in ("", None)]
        skipped = len(lineage) - len(computed)
        out.append(f"{len(computed):,} technique-years computed, {skipped:,} recorded "
                   f"as too small to compute (fewer than 3 prior or new authors). "
                   f"Skipped cells are kept in the file rather than dropped.")
        out.append("")
        ratios = [float(r["ratio"]) for r in computed]
        if ratios:
            out += [
                f"- median ratio **{statistics.median(ratios):.2f}** "
                f"(1.0 = no closer than degree-matched strangers)",
                f"- interquartile range "
                f"{_q(ratios, 0.25):.2f} – {_q(ratios, 0.75):.2f}",
                f"- share above 1.0: "
                f"{sum(1 for r in ratios if r > 1) / len(ratios):.0%}",
            ]
            out.append("")
            out += ["| technique | n years | median ratio | median z | median new authors |",
                    "|---|---|---|---|---|"]
            per: dict[str, list[dict]] = defaultdict(list)
            for row in computed:
                per[row["technique"]].append(row)
            for tid in sorted(per, key=lambda t: -statistics.median(
                    [float(r["ratio"]) for r in per[t]])):
                rows = per[tid]
                out.append(
                    f"| `{tid}` | {len(rows)} | "
                    f"{statistics.median([float(r['ratio']) for r in rows]):.2f} | "
                    f"{statistics.median([float(r['z']) for r in rows]):.2f} | "
                    f"{statistics.median([int(r['n_new_authors']) for r in rows]):.0f} |"
                )
            out += ["", "**Read the last column before the third.** Cohort size drives "
                    "the ratio toward 1 mechanically as a technique's adopter "
                    "population grows (`docs/10` §2), so this table is not a "
                    "tacitness ranking and must not be read as one."]
    else:
        out.append("_not yet computed — `scripts/build_lineage.py` is a multi-hour run_")

    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"Wrote {args.out}")
    print(f"  panel {len(panel):,} cells · corpus {len(corpus)} years · "
          f"lineage {len(lineage):,} rows")
    return 0


def _m(values: list[float]) -> str:
    return f"{statistics.mean(values):.3f}" if values else "—"


def _q(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


if __name__ == "__main__":
    raise SystemExit(main())
