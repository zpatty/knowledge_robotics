# 10 — No arbitrary classification: measure continuously

**Constraint, set by the PI (2026-08-30).** Where a quantity is continuous,
measure it continuously. Do not gate it into categories at a cut point nobody can
defend.

> I'd prefer to avoid such arbitrary limits on what classifies as
> lineage-connected. Instead, we should evaluate such things on continuous numbers
> wherever possible rather than gating them. Otherwise, we will inevitably bias our
> results.

The objection is exact, and it names a mechanism rather than a preference.
Dichotomising a continuous variable does three things, all bad:

1. **It discards information.** Co-authorship distance 2 and distance 3 are not
   different in kind. Collapsing them asserts a discontinuity in the world that
   the world does not have.
2. **It creates a researcher degree of freedom.** The cut point is chosen, and
   it moves the result. This is exactly threat **T-I** in
   [`05`](05-validation-and-threats.md), except self-inflicted and invisible —
   nothing in a results table shows that the number would differ at distance 3.
3. **It interacts with the inflation in [`09`](09-scale-invariance.md).** A fixed
   radius on a densifying graph sweeps up a growing share of the field. So a
   threshold does not merely lose information; it *manufactures a time trend*.

Points 2 and 3 compound: an arbitrary cut on an inflating quantity produces a
confident, spurious, monotone result — the most dangerous kind.

This document audits the plan's thresholds and specifies the replacement for the
worst of them.

---

## 1. Audit of thresholds in the plan

| Where | Threshold | Verdict |
|---|---|---|
| `T1` ([`02`](02-detection-methods.md) §2) | "no co-authorship path of length ≤ 2" | **Replace.** See §2. |
| `T2` ([`02`](02-detection-methods.md) §2) | "distance ≤ 2"; "first *k* adoptions" | **Replace.** Both. See §2. |
| `H4` ([`04`](04-study-design.md) §4) | "techniques within *k* years of introduction" define the frontier | **Replace** with technique age as a continuous covariate. Frontier-ness is a gradient; estimate the deficit *as a function of* age rather than splitting into frontier and mature. |
| `H5a/b` ([`04`](04-study-design.md) §4) | "restrict to techniques with ≥ 15 years of history" | **Soften.** A power requirement, not a construct claim, so it is defensible — but weight techniques by series length instead of excluding them, and report the sensitivity. |
| `04` §1 | "hand-verify the top ~500 by frequency" | **Acceptable.** A curation budget, not a measurement cut. It bounds which techniques carry quantitative claims; state it and report whether results hold across the frequency range. |
| `S2` ([`02`](02-detection-methods.md) §4) | "a small number of standard platforms" | **Already fine.** Specified as a fraction and a Herfindahl — both continuous. Do not let "small number" become a top-*n* list. |
| `V3` ([`05`](05-validation-and-threats.md)) | dose scale of "four or five ordered levels" | **Keep.** Genuinely ordinal, hand-built blind, and the monotone dose-response *is* the test. Ordered categories on a constructed scale are not the same error. |
| `B1` ([`02`](02-detection-methods.md) §6) | justification type: derived / bounded / analogical / … | **Keep.** Genuinely categorical — a proof is not a stronger version of an analogy. But record classifier confidence and propagate it rather than hard-assigning. |
| `05` | "seal a random 20% hold-out" | **Keep.** A protocol decision, not a measurement. |
| `09` (ours) | `hyper_author_threshold=20` in `corpus_baselines.py` | **Our own violation.** Replaced by reporting the author-count distribution; see §3. |
| `10` (ours) | the cross-technique probe splits cohorts at 2015 | **Our own violation**, in this document's own demo. The principled form regresses on the continuous time gap; see §2. |

The rule that separates the two columns: **a threshold on a measured quantity is a
bug; a threshold on a work plan is a budget.** Curation limits, hold-out
fractions, and annotation-sample sizes are budgets. Distance cutoffs, age splits,
and top-*k* adopter windows are measurements wearing a budget's clothes.

## 2. The replacement for T1/T2: continuous lineage proximity

Implemented in `src/tacit/lineage.py`.

**Instead of** "is this adopter within distance 2 of an originator?" — **ask** "how
close is this adopter to the originators, on a continuous scale?"

The measure is **personalised PageRank** from the originating authors over the
fractionally-weighted co-authorship graph. A walker restarts at the originators
with probability `alpha` and otherwise steps along collaboration edges; the
stationary mass on an author is their lineage proximity. Every path length
contributes, geometrically discounted, so an author at distance 4 scores lower
than one at distance 2 rather than being reclassified as a stranger.

Three properties it has that the threshold version does not:

- **No cut point.** Proximity is a real number in [0, 1]. Nobody is in or out.
- **Edge strength is carried.** Five joint papers is a stronger channel than one,
  and the weights say so instead of both being "distance 1".
- **Team size is deflated at source.** Edges are weighted 1/(*n*−1)
  ([`09`](09-scale-invariance.md) §2), so a 279-author consortium paper does not
  redefine the neighbourhood.

### `alpha` is a parameter, not a smuggled threshold

`alpha` sets how fast influence decays with distance; it does not set where
influence stops. That is a real difference, but it is still a choice, so the
protocol is: **report the whole `alpha` curve, never a single value.**
`sweep_alpha()` exists to make that the path of least resistance. A finding that
holds at α = 0.05 and dies at α = 0.30 is a finding about α.

`epsilon` in the push algorithm is numerical tolerance, not a threshold —
lowering it refines the same estimate rather than reclassifying anyone.

### Null normalisation makes it scale-free

Raw proximity still rises as the graph densifies. So the reported statistic is
**observed / expected**, where the expectation comes from the same walk run from
**degree-matched random seed sets**. Degree-matched rather than uniform, because
well-connected originators are close to everyone and a naive null would credit
their technique with transmission it never had.

A ratio of 1 means "no closer than similarly-connected strangers"; above 1 is
genuine lineage proximity. Because the null is drawn from the same graph in the
same year, density cancels — satisfying [`09`](09-scale-invariance.md) and this
document in one statistic. `z` is reported alongside, since a ratio of 1.4 against
a tight null and against a wide one are different findings.

### It works on the real corpus

Against the 68,445-author, 292,905-edge graph built from the 2006–2025 harvest,
comparing pre-2015 authors of impedance-control papers with post-2015 authors of
impedance-control papers who are not among them:

```
observed = 0.0485   expected = 0.0225   ratio = 2.15   z = 4.74
```

Later impedance-control authors sit **2.15× closer** to the earlier ones than
degree-matched strangers do. That is the transfer-channel signal T2 was reaching
for, expressed as a continuous quantity with a null attached and no cut point
anywhere in it.

And the `alpha` sweep shows why the curve is the deliverable rather than the
point:

| `alpha` | ratio | z |
|---|---|---|
| 0.05 | 1.71 | 3.05 |
| 0.15 | 2.13 | 4.33 |
| 0.30 | 2.39 | 4.28 |
| 0.50 | 2.60 | 4.34 |

The *existence* of lineage proximity is robust — every value is well above 1 and
significant against its null. Its *magnitude* is not: it rises by half again
across the range, because a larger `alpha` concentrates the walk near the
originators and so weights close collaborators more heavily. Reporting "2.15×"
alone would present a choice of `alpha` as a property of robotics. Report the
curve, and the honest claim is the one the curve supports: impedance control's
later authors are substantially closer to its earlier ones than chance, by a
factor between roughly 1.7 and 2.6 depending on how sharply influence is assumed
to decay.

*(Caveat carried from [`08`](08-phase0-findings.md) F5: with no ORCID and no
pre-2019 affiliations, authors are identified by name string alone. Name collisions
inflate connectivity and the error is worse for East Asian names — threat **T-H**.
The proximity measure inherits that error; it does not cause it, and the
degree-matched null absorbs part of it, but the disambiguation sample in
[`05`](05-validation-and-threats.md) is a precondition for trusting any absolute
level here. Ratios across techniques within the same graph are safer than levels.)*

### A cross-technique probe, and why it is not yet a finding

Running the same measure over four techniques produces an ordering that lines up
suspiciously well with the **D1** prediction in [`04`](04-study-design.md) §3 —
craft-laden techniques high, heavily-codified ones at chance:

| Technique | ratio @ α=0.15 | z | early → late authors |
|---|---|---|---|
| Visual servoing | 2.67 | 21.1 | 385 → 374 |
| Impedance control | 2.13 | 4.3 | 186 → 342 |
| Model predictive control | 1.19 | 1.4 | 98 → 776 |
| Reinforcement learning | 1.11 | 1.4 | 167 → 3,179 |

It is tempting to read this as early face validity: impedance control is
[`04`](04-study-design.md) §5's designated craft-laden case, and MPC and RL are
the two with mature solvers and enormous public codebases. **Do not report it that
way.** The right-hand column shows why.

**The two techniques at chance are exactly the two whose adopter population
exploded** — RL by 19×, MPC by 8× — while the two showing lineage proximity have
adopter cohorts that are flat or barely doubled. A cohort that grows nineteenfold
is mostly new entrants, and new entrants are unconnected to anyone, so the mean
proximity of the cohort regresses toward its null whatever the technique's tacit
content. The degree-matched null controls for how well-connected the *originators*
are; it does not control for how fast the *adopter population* grew.

This may still be the signal the transfer channel is after — broad diffusion to
strangers is what codification is supposed to look like — but it is not separable,
on this evidence, from the field-wide pivot to learning methods that inflated the
RL cohort for reasons having nothing to do with codification. That is threat
**T-G** (attention and scale confounds) in
[`05`](05-validation-and-threats.md), arriving exactly where it was predicted to.

Three things this probe therefore needs before it means anything:

1. **A growth-matched null**, or the technique's adoption trajectory as an
   explicit covariate. Comparing techniques with 19× and 1× cohort growth on an
   unadjusted ratio compares growth rates.
2. **The technique registry** ([`04`](04-study-design.md) §1). These cohorts are
   title-substring matches, not curated λ-objects; "reinforcement learning" in a
   title spans a dozen distinct techniques.
3. **A continuous treatment of time.** The probe splits cohorts at 2015 — an
   arbitrary cut point, and so a violation of this document inside its own
   demonstration. The principled form regresses proximity on the time gap between
   originating and adopting papers, with no split at all.

Recorded here rather than quietly fixed because it is a good illustration of the
constraint's value: the measure is continuous and deflated, and it *still*
produced a plausible, well-ordered, publication-shaped result that is probably an
artifact of cohort growth. Scale invariance and continuity are necessary, not
sufficient.

## 3. Our own violation, fixed

`corpus_baselines.py` shipped with `--hyper-author-threshold 20`, counting "papers
above 20 authors". That is exactly the error this document is about: 20 is
arbitrary, and a 19-author paper is not different in kind from a 21-author one.

Replaced with the **author-count distribution** — median, p90, p99, max — which
carries the same information about consortium papers without a cut point, and
shows the shape rather than a count on one side of a line. The threshold survives
only as an optional reporting convenience, never as an input to a measure.

## 4. Standing rule

**Prefer a number to a class.** Before adding a comparison, an indicator, or a
filter, ask whether the underlying quantity is continuous. If it is:

- Report it continuously, and let the model do any binning it needs.
- Where a decay or weighting parameter is unavoidable, report the sensitivity
  curve across it rather than a point estimate.
- Where a genuine category is unavoidable (`B1`'s justification types, `V3`'s
  dose scale), carry classifier confidence forward rather than hard-assigning.

This joins the three constraints in [`09`](09-scale-invariance.md) and
[`06`](06-roadmap.md) as a binding design decision. It is closely related to
[`09`](09-scale-invariance.md): scale invariance says *deflate the quantity*, this
says *do not chop it up*. A measure that violates either will produce a confident
trend that is an artifact of the instrument.
