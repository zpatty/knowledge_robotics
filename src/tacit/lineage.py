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

Pure stdlib and dict-based. The graph is ~68k authors and ~293k edges; the local
push algorithm keeps a full null sweep tractable without numpy. Phase 4's
spectral and survival work will want numpy/scipy; nothing here does.
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


def personalised_pagerank(
    graph: Graph,
    seeds: Mapping[str, float] | Sequence[str],
    *,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    max_pushes: int = 2_000_000,
    degree: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Approximate personalised PageRank by local push (Andersen-Chung-Lang).

    Power iteration touches all 68k authors every sweep and is far too slow in
    pure Python for the hundreds of runs the null model needs. Push instead
    propagates mass only where there is mass to propagate, so cost scales with
    the neighbourhood actually reached — seconds rather than minutes — with a
    provable error bound.

    **`epsilon` is a convergence tolerance that behaves as a distance cutoff when
    too loose, so it must be verified rather than assumed.** It bounds the
    residual per unit degree; work is bounded by 1/(epsilon*alpha). Measured on
    one impedance-control cell the estimate is NOT stable across it:

        epsilon=1e-6   ratio 2.48   15.7s
        epsilon=1e-5   ratio 3.73    0.9s
        epsilon=1e-4   ratio 0.00    0.1s

    At 1e-4 the walk is truncated before reaching the adopters at all, which is
    precisely the gated measurement docs/10 rules out, smuggled in through a
    performance knob. Do not loosen it for speed without re-checking convergence
    on the cells being computed. 1e-6 is the loosest value tested that still
    agrees with a tighter one; `check_convergence` re-tests that on new data.

    Pass `degree` to reuse a precomputed degree map. Recomputing it per call
    dominates the runtime when sweeping nulls, since it touches every edge.

    Returns proximity mass by author; authors never reached have no entry, which
    is a proximity of zero rather than a separate class.
    """
    if isinstance(seeds, Mapping):
        seed_weights = dict(seeds)
    else:
        seed_weights = {s: 1.0 for s in seeds}
    seed_weights = {s: w for s, w in seed_weights.items() if s in graph}
    total = sum(seed_weights.values())
    if not total:
        return {}

    if degree is None:
        degree = degrees(graph)
    estimate: dict[str, float] = {}
    residual: dict[str, float] = {s: w / total for s, w in seed_weights.items()}
    queue = list(residual)
    queued = set(queue)

    pushes = 0
    while queue and pushes < max_pushes:
        node = queue.pop()
        queued.discard(node)
        r = residual.get(node, 0.0)
        deg = degree.get(node, 0.0)
        if deg <= 0.0 or r < epsilon * deg:
            continue
        pushes += 1
        estimate[node] = estimate.get(node, 0.0) + alpha * r
        residual[node] = 0.0
        spread = (1.0 - alpha) * r
        for nbr, w in graph[node].items():
            residual[nbr] = residual.get(nbr, 0.0) + spread * (w / deg)
            if nbr not in queued and residual[nbr] >= epsilon * degree.get(nbr, 1.0):
                queue.append(nbr)
                queued.add(nbr)
    return estimate


def proximity(
    graph: Graph,
    originators: Sequence[str],
    adopters: Sequence[str],
    *,
    alpha: float = 0.15,
    epsilon: float = 1e-6,
    degree: Mapping[str, float] | None = None,
) -> float:
    """Continuous lineage proximity of an adopting group to the originators.

    The adopters' share of the personalised-PageRank mass. Continuous in [0, 1];
    no adopter is ever classified as connected or not.
    """
    rank = personalised_pagerank(
        graph, originators, alpha=alpha, degree=degree, epsilon=epsilon
    )
    if not rank:
        return 0.0
    origin = set(originators)
    # Exclude the originators themselves: the question is how close *others* are.
    mass = sum(v for k, v in rank.items() if k not in origin)
    if mass <= 0.0:
        return 0.0
    return sum(rank.get(a, 0.0) for a in adopters if a not in origin) / mass


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
    epsilon: float = 1e-6,
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
    observed = proximity(
        graph, originators, adopters, alpha=alpha, degree=deg_w, epsilon=epsilon
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
                alpha=alpha, degree=deg_w, epsilon=epsilon,
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
    originators: Sequence[str],
    adopters: Sequence[str],
    *,
    alpha: float = 0.15,
    epsilons: Sequence[float] = (1e-5, 1e-6, 1e-7),
    **kwargs,
) -> dict[float, float]:
    """Raw proximity across tolerances - run before trusting a new sweep.

    epsilon is a performance knob that silently becomes a distance cutoff when too
    loose (see personalised_pagerank). If these values disagree materially, the
    looser ones are truncating the walk rather than approximating it.
    """
    return {
        eps: proximity(graph, originators, adopters, alpha=alpha, epsilon=eps, **kwargs)
        for eps in epsilons
    }
