#!/usr/bin/env python3
"""Compare the retired push estimates against the exact solve.

Kept as evidence rather than deleted. The push implementation produced a full
panel that looked entirely reasonable — median ratio 2.58, 86% of cells above
1.0, median z 6.8 — and every one of those numbers was computed from a truncated
walk. This quantifies how wrong it was and where, which is the only way to know
whether any conclusion drawn from it survived.

Reads data/indicator_lineage_PUSH_INVALID.csv and data/indicator_lineage.csv.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path


def load(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    with path.open() as fh:
        return {
            (r["technique"], r["year"], r["alpha"]): r
            for r in csv.DictReader(fh)
            if r.get("ratio") not in ("", None)
        }


def main() -> int:
    push = load(Path("data/indicator_lineage_PUSH_INVALID.csv"))
    exact = load(Path("data/indicator_lineage.csv"))
    shared = sorted(set(push) & set(exact))
    if not shared:
        print("no overlapping computed cells yet")
        return 0

    print(f"push cells {len(push):,} · exact cells {len(exact):,} · "
          f"overlap {len(shared):,}\n")

    pr = [float(push[k]["ratio"]) for k in shared]
    er = [float(exact[k]["ratio"]) for k in shared]
    print(f"{'':<12}{'median':>9}{'mean':>9}{'share>1':>10}")
    for label, vals in (("push", pr), ("exact", er)):
        print(f"{label:<12}{statistics.median(vals):>9.2f}{statistics.mean(vals):>9.2f}"
              f"{sum(1 for v in vals if v > 1) / len(vals):>9.0%}")

    rel = [abs(p - e) / e for p, e in zip(pr, er) if e > 0]
    rel.sort()
    print(f"\nrelative error of push vs exact, over {len(rel):,} cells:")
    for q, label in ((0.5, "median"), (0.75, "p75"), (0.9, "p90"), (1.0, "max")):
        print(f"  {label:<8}{rel[min(len(rel) - 1, int(q * len(rel)))]:.0%}")
    print(f"  within 10%: {sum(1 for r in rel if r <= 0.10) / len(rel):.0%}")

    # Does the error depend on cohort size? That was the hypothesis.
    pairs = sorted(
        (int(exact[k]["n_prior_authors"]),
         abs(float(push[k]["ratio"]) - float(exact[k]["ratio"]))
         / max(float(exact[k]["ratio"]), 1e-12))
        for k in shared
    )
    q = len(pairs) // 4
    print("\nrelative error by prior-cohort quartile:")
    for i, label in enumerate(("smallest", "2nd", "3rd", "largest")):
        chunk = pairs[i * q:(i + 1) * q] if i < 3 else pairs[3 * q:]
        print(f"  {label:<10}median prior {statistics.median([c[0] for c in chunk]):>6.0f}"
              f"   median rel err {statistics.median([c[1] for c in chunk]):>7.0%}")

    # Would a conclusion drawn from push have survived?
    def spearman(xs, ys):
        def rank(v):
            order = sorted(range(len(v)), key=lambda i: v[i])
            rk = [0] * len(v)
            for pos, i in enumerate(order):
                rk[i] = pos
            return rk
        rx, ry = rank(xs), rank(ys)
        n = len(xs)
        mx, my = sum(rx) / n, sum(ry) / n
        cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
        sx = sum((x - mx) ** 2 for x in rx) ** 0.5
        sy = sum((y - my) ** 2 for y in ry) ** 0.5
        return cov / (sx * sy) if sx and sy else 0.0

    print(f"\nSpearman(push ratio, exact ratio) = {spearman(pr, er):+.3f}")
    for name, key in (("cohort growth", None), ("prior cohort size", "n_prior_authors")):
        if key:
            xs = [int(exact[k][key]) for k in shared]
        else:
            xs = [int(exact[k]["n_new_authors"]) / max(int(exact[k]["n_prior_authors"]), 1)
                  for k in shared]
        print(f"  Spearman({name}, ratio): "
              f"push {spearman(xs, pr):+.3f}   exact {spearman(xs, er):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
