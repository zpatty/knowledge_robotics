"""Continuous lineage proximity — the transfer channel without a distance cutoff.

`T1`/`T2` in docs/02 define independence by a **threshold**: an adopter is
"lineage-connected" at co-authorship distance ≤ 2. Two objections, both binding
(docs/10-continuous-measurement.md):

  1. The cutoff is arbitrary. Distance 2 and distance 3 are not different in
     kind, and the choice moves the result. Every technique sitting near the
     boundary is assigned by a decision nobody can defend.
  2. It is not scale-free. As the co-authorship graph densifies — mean degree
     rose 73% over 2006-2025 — a fixed radius sweeps up a growing share of the
     field, so a thresholded lineage ratio drifts with team size rather than
     with how knowledge moves (docs/09).

This module replaces the cutoff with a continuous score.

**Proximity.** Personalised PageRank from the originating authors. A random
walker restarts at the originators with probability `alpha` and otherwise steps
along fractionally-weighted co-authorship edges; the stationary mass on an
author is their proximity to that technique's origin. Paths of every length
contribute, weighted by a geometric decay, so nothing is gated — an author at
distance 4 scores lower than one at distance 2, rather than being reclassified.

**`alpha` is a parameter, not a threshold, and is never fixed.** It sets how
fast influence decays with distance rather than where it stops. Report the whole
`alpha` curve; a finding that survives only at one value is an artifact of that
value. `sweep_alpha` exists to make that the default way of using this.

**Null normalisation.** Raw proximity still rises with graph density, so it is
reported against a degree-preserving null: the same walk from seed sets matched
on degree, drawn at random. `proximity_vs_null` returns observed / expected,
which is both continuous and scale-free — satisfying docs/09 and docs/10 in one
statistic. A value of 1 means "no closer to the originators than chance given
how well-connected these people are"; above 1 means genuine lineage proximity.

Pure stdlib and dict-based. The graph is ~68k authors and ~293k edges. The
measure is computed by one linear solve per adopter set (`absorption`), reused
across the observed value and every null draw, which is both exact and cheaper
than a PageRank per seed set. Phase 4's spectral and survival work will want
numpy/scipy; nothing here does.
"""
from __future__ import annotations

import itertools
import random
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

Graph = dict[str, dict[str, float]]


def degree_buckets(graph: Graph) -> dict[int, list[str]]:
    """Authors grouped by unweighted degree, for degree-matched null sampling.
    Built once and shared: rebuilding it per call dominates a panel-wide sweep."""
    buckets: dict[int, list[str]] = defaultdict(list)
    for node, nbrs in graph.items():
        buckets[len(nbrs)].append(node)
    return buckets


def degrees(graph: Graph) -> dict[str, float]:
    """Weighted degree per author. Computed once and threaded through: it touches
    every edge, which dominates runtime if recomputed inside a null sweep."""
    return {node: sum(nbrs.values()) for node, nbrs in graph.items()}


def build_coauthor_graph(
    papers: Iterable[Mapping],
    *,
    fractional: bool = True,
) -> Graph:
    """Weighted co-authorship graph keyed on author name.

    Edges are fractionally weighted by default: each author of an n-author paper
    contributes one unit of collaboration, so an edge carries 1/(n-1). Without
    this a single 279-author consortium paper contributes ~39,000 edges and
    dominates the year (docs/09 §2). `fractional=False` exists only so the
    difference can be measured, not as a supported analysis mode.

    Weights from repeated collaborations accumulate: working together five times
    is stronger evidence of a transmission channel than working together once,
    and that is a continuous fact the graph should carry rather than round off.
    """
    graph: Graph = defaultdict(dict)
    for paper in papers:
        names = sorted({
            a["name"] for a in (paper.get("authors") or []) if a.get("name")
        })
        if len(names) < 2:
            continue
        weight = 1.0 / (len(names) - 1) if fractional else 1.0
        for u, v in itertools.combinations(names, 2):
            graph[u][v] = graph[u].get(v, 0.0) + weight
            graph[v][u] = graph[v].get(u, 0.0) + weight
    return dict(graph)


def absorption(
    graph: Graph,
    targets: Sequence[str],
    *,
    alpha: float = 0.15,
    tol: float = 1e-10,
    max_iter: int = 400,
    degree: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Solve x = 1_S + (1-alpha) P x, where P is the degree-normalised walk.

    This is the whole measure, turned inside out. The quantity wanted is the
    personalised-PageRank mass that a seed set T puts on an adopter set S:

        pi_T . 1_S  =  alpha * mean over u in T of x(u)

    where x depends only on S. So **one solve serves the observed value and every
    null draw**, because the nulls differ only in T. That is a 6x saving over
    running a separate PageRank per seed set, and it is exact.

    It also deletes a parameter. The previous implementation was an approximate
    local push with an `epsilon` tolerance and a `max_pushes` cap, and it was
    wrong: at tight tolerances it hit the cap and returned partial results, so
    values moved non-monotonically (0.000124 -> 0.000372 -> 0.000026 across
    1e-7, 1e-8, 1e-9) and small-cohort cells disagreed with a tighter run by up
    to 100%. A truncated walk is a distance cutoff, which docs/10 forbids, so
    there was no tolerance at which that implementation was both fast and
    honest.

    Power iteration here has one convergence criterion, no cap, and no knob that
    silently changes the answer: it runs until the update is below `tol`.
    Convergence is geometric at rate (1-alpha), so the default reaches ~1e-10 in
    about 140 sweeps.
    """
    if degree is None:
        degree = degrees(graph)
    target_set = {t for t in targets if t in graph}
    if not target_set:
        return {}

    decay = 1.0 - alpha
    x: dict[str, float] = {t: 1.0 for t in target_set}
    for _ in range(max_iter):
        nxt: dict[str, float] = {t: 1.0 for t in target_set}
        for node, value in x.items():
            if value == 0.0:
                continue
            share = decay * value
            for nbr, w in graph[node].items():
                d = degree.get(nbr, 0.0)
                if d > 0.0:
                    nxt[nbr] = nxt.get(nbr, 0.0) + share * (w / d)
        delta = max(
            abs(nxt.get(k, 0.0) - x.get(k, 0.0)) for k in set(nxt) | set(x)
        )
        x = nxt
        if delta < tol:
            break
    return x


def proximity(
    graph: Graph,
    originators: Sequence[str],
    adopters: Sequence[str],
    *,
    alpha: float = 0.15,
    degree: Mapping[str, float] | None = None,
    solution: Mapping[str, float] | None = None,
) -> float:
    """Personalised-PageRank mass the originators place on the adopters.

    Continuous in [0, 1]; every path length contributes under geometric decay and
    no adopter is ever classified as connected or not.

    Pass `solution` (from `absorption` on the adopter set) to reuse the solve
    across seed sets — that is what makes the null sweep affordable.
    """
    if solution is None:
        solution = absorption(graph, adopters, alpha=alpha, degree=degree)
    if not solution:
        return 0.0
    seeds = [u for u in originators if u in graph]
    if not seeds:
        return 0.0
    return alpha * sum(solution.get(u, 0.0) for u in seeds) / len(seeds)


def _degree_matched_seeds(
    graph: Graph,
    originators: Sequence[str],
    rng: random.Random,
    buckets: Mapping[int, Sequence[str]],
    degree_of: Mapping[str, int],
) -> list[str]:
    """Random seeds with the same degree profile as the originators.

    Degree-matched rather than uniform: well-connected originators are close to
    everyone, and a null that ignored that would credit their technique with
    lineage transmission it did not have.
    """
    out: list[str] = []
    keys = sorted(buckets)
    for author in originators:
        d = degree_of.get(author)
        if d is None:
            continue
        # Nearest populated degree bucket, so rare degrees still get a match.
        best = min(keys, key=lambda k: (abs(k - d), k))
        out.append(rng.choice(buckets[best]))
    return out


def proximity_vs_null(
    graph: Graph,
    originators: Sequence[str],
    adopters: Sequence[str],
    *,
    alpha: float = 0.15,
    n_null: int = 50,
    seed: int = 0,
    degree: Mapping[str, float] | None = None,
    degree_buckets: Mapping[int, Sequence[str]] | None = None,
) -> dict[str, float]:
    """Observed proximity over its degree-matched expectation.

    The scale-free form of the measure. `ratio` near 1 means the adopters are no
    closer to the originators than similarly-connected strangers would be; above
    1 is genuine lineage proximity. Because it is a ratio against a null drawn
    from the *same* graph, it does not inflate as the graph densifies, which is
    what a raw proximity would do (docs/09).

    `z` is reported alongside because the null spread matters: a ratio of 1.4 on
    a tight null and on a wide one are different findings.
    """
    deg_w = degree if degree is not None else degrees(graph)
    # One solve, reused by the observed value and every null draw.
    solution = absorption(graph, adopters, alpha=alpha, degree=deg_w)
    observed = proximity(
        graph, originators, adopters, alpha=alpha, degree=deg_w, solution=solution
    )

    degree_of = {node: len(nbrs) for node, nbrs in graph.items()}
    if degree_buckets is None:
        buckets: dict[int, list[str]] = defaultdict(list)
        for node, d in degree_of.items():
            buckets[d].append(node)
    else:
        buckets = degree_buckets

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_null):
        null_seeds = _degree_matched_seeds(
            graph, originators, rng, buckets, degree_of
        )
        if null_seeds:
            draws.append(proximity(
                graph, null_seeds, adopters,
                alpha=alpha, degree=deg_w, solution=solution,
            ))

    if not draws:
        return {"observed": observed, "expected": 0.0, "ratio": 0.0, "z": 0.0}

    expected = sum(draws) / len(draws)
    var = sum((d - expected) ** 2 for d in draws) / len(draws)
    sd = var ** 0.5
    return {
        "observed": observed,
        "expected": expected,
        "ratio": observed / expected if expected else 0.0,
        "z": (observed - expected) / sd if sd else 0.0,
        "n_null": len(draws),
    }


def sweep_alpha(
    graph: Graph,
    originators: Sequence[str],
    adopters: Sequence[str],
    *,
    alphas: Sequence[float] = (0.05, 0.10, 0.15, 0.25, 0.40, 0.60),
    **kwargs,
) -> dict[float, dict[str, float]]:
    """The measure across the decay range — the intended way to report it.

    A result that holds only at one `alpha` is a result about `alpha`. Publish
    the curve; if it is flat, the finding is robust to how fast influence is
    assumed to decay, and that robustness is itself the claim.
    """
    return {
        a: proximity_vs_null(graph, originators, adopters, alpha=a, **kwargs)
        for a in alphas
    }


def check_convergence(
    graph: Graph,
    targets: Sequence[str],
    *,
    alpha: float = 0.15,
    tols: Sequence[float] = (1e-8, 1e-10, 1e-12),
    **kwargs,
) -> dict[float, float]:
    """Total solved mass across tolerances - a convergence check with teeth.

    Kept after the push implementation was replaced, because "it converged" is a
    claim that should be re-tested on new data rather than assumed. Values here
    should agree to several digits; if they do not, `max_iter` is binding.
    """
    return {
        tol: sum(absorption(graph, targets, alpha=alpha, tol=tol, **kwargs).values())
        for tol in tols
    }
