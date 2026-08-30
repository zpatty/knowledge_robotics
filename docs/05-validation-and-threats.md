# 05 — Validation and threats to validity

The project's central risk is not that the analysis will be wrong but that the *measures*
will be vacuous — that we produce a number, call it tacitness, and it turns out to track
writing style or venue page limits. Validation is therefore not a final step; it is the
gating condition on everything in [`04`](04-study-design.md).

## 1. Validation strategy — five independent lines

No single validation is sufficient. Convergence across independent lines is the argument.

### V1 — Known-case validation
Assemble a set of techniques whose tacit status is independently known from the historical
and STS record, plus robotics folklore: techniques that famously did not transfer, and
techniques that transferred instantly. Do the measures separate them? This is coarse and
small-*n*, but it is the cheapest thing that could falsify the whole battery, so do it first.

### V2 — Expert elicitation
Survey 30–60 active roboticists. For a stratified sample of ~50 technique-year pairs, ask:
*if you had only this paper, could your lab reproduce this? What would you need to ask the
authors? Would you expect to need to visit or hire someone?* Collect ratings and free text.

This is the study's primary construct-validity anchor and should be treated as a first-class
component, not an afterthought: pre-register the instrument, get IRB approval early
(it is minimal-risk but the lead time is real), and recruit through the community rather
than convenience-sampling one department. Correlate expert ratings against every indicator.
An indicator that does not correlate with expert judgment needs either a defense or a
deletion.

**Bonus design:** ask respondents about techniques *from their own subfield* and *from
outside it*. The gap between insider and outsider assessments of the same paper is a direct
measurement of collective tacit knowledge (CTK) — the thing our text instruments cannot
see. That is a small, clean, self-contained result worth publishing on its own.

### V3 — Reproduction ground truth
- **F1 extractions**: papers that state they failed to reproduce something give us real
  labels, free.
- **RA-L reproducible-article track** and any robotics reproducibility challenges: papers
  vetted as reproducible are a positive class.
- **Raff (2019)** and successor ML reproducibility datasets: out-of-domain but directly
  comparable labels; if our indicators predict his outcomes, that is strong external
  validity.
- **Reproducibility challenges / student reimplementation exercises**: if the project has
  access to a robotics course, assigning reimplementations of sampled papers generates
  ground truth at low cost and is pedagogically defensible on its own terms.

### V4 — The codification-event test
Predictive validity through an event study: indicators should *drop discontinuously* at
codification events, and drop more for events that codify more (a full pipeline release
should move them more than a partial one; a standard platform more than a one-off script).
A measure that does not respond to a known code release is not measuring codification.
Dose-response here is a strong test.

### V5 — Placebo and negative controls
- **Style placebo.** Construct a measure from purely stylistic features with no theoretical
  relation to tacitness (sentence length, passive voice, readability). If it reproduces our
  headline results, our results are about writing, not knowledge.
- **Field control.** Apply the battery to a venue where the answer is known a priori — a
  theory venue (should be low-deficit; the proof is the artifact) and a wet-lab biology or
  materials venue (should be high). If robotics does not land between them in the expected
  place, stop.
- **Shuffled-lineage control.** For transfer-channel results, permute the co-authorship
  graph and confirm the lineage effect vanishes.

## 2. ESP-specific validation

The novel instrument gets the heaviest scrutiny:

| Check | Design | Pass criterion |
|---|---|---|
| **Ablation sensitivity** | Delete a known specification element (a stated hyperparameter, a preprocessing step) from a paper; re-run ESP-B. | Divergence rises, monotonically with amount deleted. |
| **Enrichment sensitivity** | Splice in the corresponding detail from the released code. | Divergence falls. |
| **Paraphrase invariance** | Rewrite equally-specific text with different surface form. | Divergence unchanged. |
| **Cross-model agreement** | ≥ 2 model families, different training data. | High rank correlation. |
| **Contamination** | Recognition probe; pre/post-cutoff strata. | Results hold in the uncontaminated stratum. |
| **Human concordance** | Experts (V2) attempt the same enumeration on ~30 papers. | Model gap-lists overlap substantially with human ones. |

Ablation + enrichment together are the decisive pair: they establish that the measure
responds to *specification content* and not to prose. **If they fail, ESP is dropped and
the study proceeds on the transfer, failure, and substitution channels** — which is why
the roadmap keeps those independent of it.

## 3. Threats to validity

### T-A — Writing-convention drift *(highest severity for linguistic indicators)*
Forty years of changing page limits, template rules, review norms, and — since ~2023 —
LLM-assisted writing. Any linguistic time trend is confounded with all of these.
**Mitigations:** within-year comparisons as the default; venue-policy fixed effects; a
hand-compiled timeline of page-limit and supplementary-policy changes; a post-2022 sensitivity
analysis; and the style placebo (V5). **Rule: no headline claim rests on a linguistic
indicator's raw time trend.** Cross-sectional and event-study designs are safe; long-run
linguistic trends are not.

### T-B — Openness selection
Covered in [`03`](03-corpus.md) §3.3. Restated because it is the most likely source of a
wrong *published* number.

### T-C — Circularity
If tacitness is defined partly by code availability and validated against code availability,
we have proved nothing. **Rule:** every validation must use a data source disjoint from the
indicators being validated. Keep a dependency matrix of indicator → data source and check it
before each validation claim.

### T-D — Survivorship
Published papers are successful projects. The tacit knowledge that killed a project never
enters the corpus at all, so the deficit we measure is a lower bound and is biased toward
techniques that worked. Partially addressable through negative results in ablation sections
and workshop papers; mostly it is a limitation to state, and it points at where the
qualitative work (V2 interviews) earns its keep.

### T-E — Technique identity instability
Discussed in [`04`](04-study-design.md) §1. Report all key results at two granularities.

### T-F — Uncodified ≠ tacit
Cowan, David & Foray's objection: most of what we detect may be knowledge that *could* have
been written down cheaply and simply wasn't. This is not a flaw to hide but the study's most
interesting internal distinction, and the Collins taxonomy exists to handle it. Concretely:
knowledge that gets codified shortly after we flag it was RTK (and the flag was correct);
knowledge that resists codification across repeated attempts is a candidate for STK/MTK.
**Persistence under codification pressure becomes an empirical criterion for separating the
kinds** — which turns the objection into a measurement.

### T-G — Attention/scale confounds
Big, fashionable techniques diffuse fast for reasons unrelated to tacitness. Everything in
the transfer channel must control for attention (citations, papers/year, funding), and
prefer within-technique designs.

### T-H — Author disambiguation error
Systematically worse for non-Western names ([`03`](03-corpus.md) §3.4). Report the error
rate broken out by name origin, and run the transfer-channel results on the ORCID-only
subset as a robustness check.

### T-I — Analyst degrees of freedom
The indicator battery is large and the hypothesis set is rich; the temptation to fish is
correspondingly large. **Pre-register** the hypotheses in [`04`](04-study-design.md) §4 and
the primary indicator specification before running the analysis; keep a hold-out slice of
the corpus (e.g. a random 20% of techniques) sealed until the specification is fixed;
report all pre-registered results including nulls.

## 4. Stopping conditions

State in advance what would make us abandon or restructure, so that the decision is not made
under sunk-cost pressure:

- **D1 fails** — the subfield gradient does not appear. The instrument is not measuring
  tacitness. Stop and rebuild.
- **ESP ablation/enrichment fails** — drop ESP, continue on the behavioral channels.
- **Expert correlation (V2) is near zero across all channels** — the construct is not
  recoverable from text. This is itself a publishable negative result *if* the validation
  design is strong enough to support it, which is a further reason to build V2 early rather
  than late.
- **Full text unobtainable** — pivot to a metadata-only study on the transfer and
  substitution channels. Narrower, still novel, still Mokyrian.
