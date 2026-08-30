# 04 — Study design

## 1. Unit of analysis: the technique, not the paper

Knowledge does not live in documents; it lives in techniques that persist across documents.
The study therefore builds a **technique registry** — a curated set of λ-objects tracked
across the corpus — and papers become *observations* of techniques rather than the units
themselves. This one decision is what makes the project a study of knowledge instead of a
study of writing.

**Building the registry.** Hybrid, in three passes:
1. *Seed by hand* — ~150 techniques chosen to span the algorithm↔hardware axis and the
   whole time range, with dated origin papers.
2. *Expand automatically* — mine method-name mentions (capitalized method names, acronyms
   with expansions, "the X method/algorithm/controller"), cluster aliases, and link to
   origin papers via citation patterns.
3. *Curate* — hand-verify the top ~500 by frequency; these carry the quantitative results.
   The long tail supports corpus-wide descriptive statistics only.

**Technique identity is the hard part.** Is "domain randomization" one technique or a
family? Is impedance control in 1990 the same object as in 2020? Adopt an explicit rule —
a technique is one λ-node if a practitioner would say they are doing "the same thing," and
resolve borderline cases by recording *both* a coarse family node and fine-grained variant
nodes, then reporting whether results are stable across the two granularities. Where they
are not stable, say so.

## 2. Data model

Five entity types and the edges among them. Sketch, not final schema:

```
Paper      (id, venue, year, pages, sections, has_video, has_code, platform[], …)
Technique  (λ)  (id, name, aliases[], family, origin_paper[], first_year, subfield)
Principle  (Ω)  (id, statement, domain, source[], formality)
Artifact        (id, kind ∈ {code, dataset, benchmark, platform, tutorial, standard,
                 textbook, competition}, release_date, technique[], completeness)
Actor           (id, orcid, affiliation_history[], advisor_of[], coauthor_edges[])

Edges
  Paper      --introduces/uses/extends/compares/reproduces-->  Technique
  Technique  --rests_on (epistemic base)                   -->  Principle
  Technique  --codified_by                                  -->  Artifact
  Actor      --authored-->  Paper ;  Actor --moved_to--> Institution @ t
  Paper      --cites (with intent label)                    -->  Paper
```

The **λ → Ω edge is Mokyr's epistemic base made explicit and computable.** Its in-degree,
diversity, and formality give B(λ,t) from [`01`](01-framework.md) §1.4.

**Main analysis table:** a technique × year panel carrying every indicator from
[`02`](02-detection-methods.md), plus adoption counts, adopter-lineage composition,
codification-event flags, and performance-trajectory data where a benchmark exists.

## 3. Descriptive questions (answer these before any causal claim)

- **D1.** How does the codification deficit distribute across robotics subfields? Prediction:
  a clean gradient from estimation/control (low) through planning and perception to
  manipulation, soft robotics, and fabrication (high). If this gradient does not appear, the
  instrument is broken and everything downstream is void — **this is the primary
  face-validity test of the entire battery.**
- **D2.** How has it moved over 40 years — in aggregate, within subfield, and within
  technique?
- **D3.** Do the channels agree? Where they diverge (e.g. a subfield with low linguistic
  tacitness but high transfer tacitness), what characterizes it? Divergence is a finding.
- **D4.** What is the distribution of epistemic-base width, and how does it relate to the
  deficit cross-sectionally?
- **D5.** Where does craft advice live, and where has it migrated to?

## 4. Hypotheses

Each is stated with its Mokyrian rationale, the identification strategy, and — importantly —
what would falsify it.

### H1 — Epistemic base width predicts sustained improvement
*Mokyr:* techniques with narrow bases can be executed but not systematically improved;
progress stalls once trial-and-error exhausts the neighborhood.
*Test:* on techniques with a benchmark trajectory, regress the duration and slope of
performance improvement on B(λ,t₀), controlling for subfield, attention (papers/year), and
compute trend.
*Falsified if:* narrow-base techniques improve as long and as fast as wide-base ones — which
in the deep-learning era is a live possibility and would be a genuinely interesting result
against Mokyr, since scale may substitute for understanding.

### H2 — Codification events cause independent diffusion *(the headline causal claim)*
*Mokyr:* falling access costs widen the community that can use a technique.
*Test:* **difference-in-differences** around codification events (S4), with the outcome being
the lineage composition of adopters (T2) and the arrival rate of lineage-unconnected
adopters. Treated: techniques receiving a code/benchmark/platform release. Control: matched
techniques on pre-trend adoption, subfield, age, and citation trajectory.
*Endogeneity:* releases are not random — authors release when a technique is succeeding.
Handle with (a) matching on pre-trends and an explicit pre-trend test, (b) instrumenting on
*exogenous* release timing where it exists (funder mandates, venue artifact-evaluation
policies, lab-wide release policies affecting techniques of varying maturity), (c) the
sharper within-technique design below.
*Sharper variant:* compare *the same technique* across adopter populations that differ in
access to the artifact — e.g. before/after a repo becomes public versus available-on-request,
or across language/region groups differing in access to an English-language tutorial.

### H3 — Codification decentralizes practice
*Mokyr:* access costs are geographic and institutional as much as intellectual.
*Test:* after codification events, does the institutional and geographic Herfindahl index of
a technique's practitioners fall? Does adoption reach outside elite institutions faster?
This is the most policy-relevant result the study can produce.

### H4 — The tacit frontier is conserved, not eliminated *(the most interesting hypothesis)*
*Mokyr, extended:* codification does not shrink the tacit stock; it moves it. Every
codification of a layer creates new practice on top of it whose own craft is uncodified. ROS
codified middleware — and the craft moved to launch-file configuration, then to reward
shaping, then to data curation.
*Test:* measure total codification deficit at the *frontier* (techniques within *k* years of
introduction) separately from the deficit in mature techniques. Prediction: mature-technique
deficit declines steadily; frontier deficit is roughly stationary across 40 years. Also test
the displacement claim directly by tracking *what kind* of craft advice (R1 classes) is
dominant in each era.
*Falsified if:* frontier deficit trends down — which would mean the field is genuinely
becoming more explicable, a strong claim in either direction.

### H5 — Direction of travel between Ω and λ
*The question the user is really asking.* Does theory precede and enable codification, or
does practice run ahead and theory arrive afterwards to explain it? Mokyr's history contains
both (the steam engine long preceded thermodynamics; the transistor did not precede solid-state
physics).
*Test:* **cross-lagged panel / panel VAR** on the technique × year panel with B(λ,t) and
D(λ,t) as the two series. Estimate lead–lag in both directions, by subfield and by era.
*Expected texture:* control and estimation should show Ω→λ; deep-learning-era manipulation
and locomotion should show λ→Ω, with theory chasing practice. If robotics has shifted from
the first regime to the second over 40 years, that is the study's most quotable finding.

### H6 — An embodiment floor
*Test:* decompose the deficit into components attributable to algorithmic versus physical
content (using S1/S2/F3 and hardware-technique flags). Prediction: the algorithmic component
declines sharply with codification; the physical component has a floor that standard
platforms *substitute for* rather than remove — visible as a deficit that drops
discontinuously at platform adoption but does not trend down otherwise.

### H7 — Narrow bases and churn
*Test:* do narrow-base subfields show higher rates of reversal, non-replication (F1), and
short-lived techniques? Connects directly to the reproducibility literature and gives the
study an audience beyond history-of-technology.

## 5. Case studies

Corpus-scale measurement without close reading produces confident nonsense. Pair every
quantitative result with ~6 deeply-traced techniques spanning the axis, each read
qualitatively end to end and used to check that the indicators are tracking what we think:

| Technique | Why chosen |
|---|---|
| ICP / point-cloud registration | Long-lived, mathematically grounded, heavily codified. Low-deficit anchor. |
| RRT / PRM / sampling-based planning | Theory-first, near-total codification via OMPL. Tests H2 with a clean event. |
| Visual servoing | Spans decades; wide base; interesting hardware dependence. |
| Impedance / compliance control | Textbook Ω, notoriously craft-laden λ. The clearest RTK/STK split in the field. |
| Sim-to-real RL for legged locomotion | The field's own explicit tacit-residual discourse; recent, well-documented, code-rich. |
| Soft-robot fabrication | Highest craft content; near-pure STK/MTK; the high-deficit anchor. |

Plus **ROS itself** as an infrastructure case — not a technique but a codification
institution, and the closest robotics analogue to the Royal Society and the technical
encyclopedia in Mokyr's account.

## 6. What "done" looks like

Two papers, plus a dataset:

- **Paper 1 (methods).** "Measuring the codification deficit in an engineering literature."
  The channel framework, the ESP instrument, validation. Venue: a scientometrics or
  computational-social-science venue, or *Research Policy*.
- **Paper 2 (findings).** "Tacit knowledge and the epistemic base in forty years of
  robotics." H2, H4, H5 as the spine. Venue: *Research Policy* / *Science and Public Policy*
  / an STS venue; possibly a robotics venue for the reflexive audience, which would be a
  more interesting choice.
- **Dataset release.** Technique registry, codification-event registry, indicator panel.
