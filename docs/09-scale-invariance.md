# 09 — Scale invariance: no indicator may rest on a raw count

**Constraint, set by the PI (2026-08-30).** Every indicator must be invariant to the
size of the thing it is measured on. A metric that depends on a raw count — of
references, papers, authors, adoptions, or events — is measuring the growth of the
corpus, not the codification deficit.

The PI's framing, which is the right one and should survive into the paper:

> If anything we do rests on raw citation count, we have failed in our mission
> anyway. Our metrics should not be dependent on raw numbers like that, in the same
> way that it would be stupid to draw economic conclusions from 2006 dollars from
> then until now.

This is a *nominal versus real* problem, and it is the same problem in every place
it appears. The corpus has inflated on several dimensions at once, and any measure
quoted in nominal units will drift for reasons that have nothing to do with what
roboticists know.

This document states the deflators, audits the battery in
[`02`](02-detection-methods.md) against them, and records where the constraint is
already violated.

---

## 1. The measured inflation

From the harvested corpus (40,293 papers, 2006–2025, `scripts/corpus_baselines.py`):

| Dimension | 2006 | 2025 | Factor |
|---|---|---|---|
| Papers per year | 1,029 | 3,489 | **×3.4** |
| References per paper (mean) | 14.6 | 33.9 | **×2.3** |
| References per paper (median) | 14 | 32 | ×2.3 |
| Authors per paper | 3.39 | 5.18 | **×1.5** |
| Distinct authors per year | 2,720 | 12,989 | ×4.8 |
| Co-authorship mean degree (raw) | 3.78 | 6.54 | **×1.7** |
| Co-authorship mean degree (fractional) | 1.27 | 1.39 | ×1.1 |

Mean and median references track each other closely, so this is a genuine shift in
norms rather than a few outliers pulling an average. Three separate inflations —
corpus size, reference-list length, and team size — compound.

## 2. The consequence that is easy to miss: the graph densifies

Reference and author inflation are visible. Their effect on the **co-authorship
graph** is not, and it lands on the study's strongest indicator family.

`T2` (lineage ratio) classifies an adoption as lineage-connected at co-authorship
distance ≤ 2. As the graph densifies, the share of the field within distance 2 of
any given author rises **mechanically**. Mean degree rose 3.78 → 6.54 across the
window — 73% — before any change in how knowledge actually moves. A
naively computed lineage ratio would therefore trend upward across the window and
be read as knowledge becoming *more* dependent on personal contact, when the only
thing that changed is that robotics papers now have more authors.

`T1` (independent adoption latency) inherits the same problem through its
independence criterion.

**This is the single most important instance of the constraint**, because T1/T2 are
described in [`02`](02-detection-methods.md) §2 as the strongest indicator family
and they carry H2, the headline causal claim.

### Hyper-authored papers break it outright

One paper — *Open X-Embodiment* (2024, **279 authors**) — contributes about 39,000
of that year's ~73,000 co-authorship edges, more than doubling the graph on its
own. Mean degree for 2024 reads 12.67 against 5.66 in 2023 and 6.54 in 2025.

(Counting convention, since it changes the numbers: a co-authorship graph counts
each *pair* once per year however often the two publish together. Counting repeat
collaborations instead would fold publication volume — itself an inflating
quantity — into a measure of who works with whom.)

A single consortium paper should not be able to redefine what "lineage-connected"
means for a whole year. Any co-authorship measure must therefore be **fractionally
weighted** — each author of an *n*-author paper contributing one unit of
collaboration, so each edge carries weight 1/(*n*−1) — rather than treating every
co-authorship as equivalent.

**This works, and the size of the correction is the argument for the whole
document.** `scripts/corpus_baselines.py` computes both:

| Measure | 2006 | 2025 | Inflation |
|---|---|---|---|
| Co-author mean degree, **raw** | 3.78 | 6.54 | **×1.73** |
| Co-author mean degree, **fractional** | 1.27 | 1.39 | **×1.09** |

Raw degree inflates 73% across the window. Fractionally weighted, it inflates 9% —
and that residual is plausibly a real signal about collaboration rather than an
artifact of team size. The *Open X-Embodiment* year shows the same thing locally:
2024 reads **13.32** raw against 5–6 in every neighbouring year, and **1.40**
fractional, in line with 1.32 (2023) and 1.39 (2025). The deflator removes the
anomaly without special-casing the paper.

An unweighted lineage ratio would have carried a 73% trend that has nothing to do
with tacit knowledge, in the indicator family the study leans on hardest.

## 3. Audit of the indicator battery

Against the constraint. "Ratio" means already scale-free by construction; "deflate"
means the specification is a rate but the denominator is unstated or wrong;
"breaks" means it is currently a raw count.

| Indicator | Status | Note |
|---|---|---|
| `R1` craft-advice density | ratio | Rate per sentence. Already normalised within year × subfield × page-count strata. |
| `R2` codification locus | ratio | A distribution over positions. Compositional, safe. |
| `R3` unexplained-parameter density | ratio | bare / total. Safe. |
| `R4` acknowledgment structure | deflate | "Per paper" is right, but acknowledgment *length* has its own trend — normalise within it. |
| `T1` adoption latency | **deflate** | A duration, but the independence test rests on the densifying graph. See §2. |
| `T2` lineage ratio | **breaks** | Distance-≤2 on an unweighted, densifying graph. Needs fractional edge weights and a per-year null. See §2. |
| `T3` citation/implementation divergence | ratio | use / total citations. **The model case** — inherently deflated. |
| `T4` personnel-flow diffusion | deflate | Hazard model; must carry per-year baseline mobility, not raw arrivals. |
| `F1` reproduction failures | **breaks** | Currently a count of extracted hits. Must be per paper, per year — raw hits grow ×3.4 with the corpus alone. |
| `F2` reimplementation gap | ratio | A performance difference; report relative, not absolute. |
| `F3` sim-to-real gap | ratio | Safe if expressed as a ratio of the two figures. |
| `F4` trial/reset disclosure | ratio | Presence rate per paper. Safe. |
| `S1` carrier portfolio | ratio | Presence rate per paper. Safe. |
| `S2` platform concentration | ratio | A fraction, and a Herfindahl. Safe. |
| `S3` infrastructure depth | **breaks** | Specified as "a count *and* a fraction". The count half must go, or become depth-per-pipeline-stage. |
| `S4` codification events | **breaks** | Event counts rise with field size. Needs events per technique per year, or per paper. |
| `U1` checklist completeness | ratio | Fraction of required elements present. Safe. |
| `U2` delegation density | deflate | Chain *depth* is scale-free and safe. *Density* needs an explicit denominator. |
| `U3` hedge attachment | deflate | Count of attached hedges needs normalising by specification elements, not by document length. |
| `B1` justification type | ratio | Categorical distribution. Safe. |
| `B2` reference reach | ratio | Fraction of citations outside robotics — **already correctly specified**, and it is load-bearing for H5. But see below. |
| `B3` explanatory vocabulary | ratio | Ratio by construction. Safe. |
| `B4` operating envelope | ratio | Presence. Safe. |

**Four indicators break outright** (`T2`, `F1`, `S3`, `S4`) and five need their
denominator stated (`R4`, `T1`, `T4`, `U2`, `U3`). None of this is hard to fix, but
all of it has to be fixed before any of them is computed, because a deflator applied
after the fact to a stored raw count is not recoverable if the count was aggregated.

### B2 is deflated but not immune

`B2` is a fraction, so reference-list inflation does not move it directly. Two
second-order effects remain and should be checked rather than assumed away:

- **The age distribution** is part of B2's specification, and a longer reference list
  is not a uniformly stretched one — recent citations grow faster than old ones, so
  the age mix shifts even at constant "reach".
- **Composition.** Longer lists plausibly add related-work citations (internal to
  robotics) faster than foundational ones (external), which would push B2 down over
  time for bibliographic reasons rather than epistemic ones.

Since B2 is the only epistemic-base indicator computable across the full window and
therefore carries H5, both effects need an explicit test, not a footnote.

## 4. Standing rule

**Every indicator ships with its denominator.** A pull request adding an indicator
states, in the code, what it is normalised by and why that denominator is the right
one. An indicator whose value would change if the corpus doubled in size, with no
change in practice, is not finished.

This joins the two constraints in [`06`](06-roadmap.md) — no LLM-based text
assessment, no expert survey — as a binding design decision rather than a
preference. It connects directly to threat **T-A** (writing-convention drift) in
[`05`](05-validation-and-threats.md): reference-list growth and team-size growth are
exactly the venue-norm changes T-A warns about, now measured rather than suspected.
