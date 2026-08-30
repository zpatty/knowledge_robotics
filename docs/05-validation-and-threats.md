# 05 — Validation and threats to validity

The project's central risk is not that the analysis will be wrong but that the *measures*
will be vacuous — that we produce a number, call it tacitness, and it turns out to track
writing style or venue page limits. Validation is therefore not a final step; it is the
gating condition on everything in [`04`](04-study-design.md).

## V0 — What validation is available, after the survey was tabled

The expert elicitation survey was the design's primary **construct-validity** anchor: the
one line connecting our indicators to what a practitioner means by "you would have to visit
their lab". It is deferred ([`appendix-deferred.md`](appendix-deferred.md) §A), and this
document is written around its absence rather than pretending otherwise.

What that costs, stated plainly, because it should appear in the papers' limitations
section in roughly this form:

- **No direct construct validity.** Nothing left connects an indicator to a human judgment
  of the construct. Everything below is predictive validity (V3), face validity (V1),
  internal consistency (V5), or negative controls (V4).
- **No units, so no levels.** Every claim must be comparative — a ranking, a gradient, an
  event-study delta, a phase relationship. **No claim of the form "robotics is X% tacit"**,
  and no absolute threshold anywhere in the analysis.
- **Collective tacit knowledge is out of scope.** The insider/outsider design was the only
  instrument that reached it. U2's unnamed-convention class is a lexical hook, not a
  measurement, and should be presented as a descriptive curiosity rather than a CTK result.

What carries the load instead — and the honest summary is that **V3 (codification-event
dose-response) is now the primary anchor**, because it is the only remaining line with real
predictive force. If a measure does not move when a technique is demonstrably codified, it
is not measuring codification, and no amount of internal consistency rescues it.

## V1 — Known-case validation
Assemble a set of techniques whose tacit status is independently known from the historical
and STS record, plus robotics folklore: techniques that famously did not transfer, and
techniques that transferred instantly. Do the measures separate them? Coarse and small-*n*,
but it is the cheapest thing that could falsify the whole battery, so do it first.

Do this **blind**: have the domain roboticist classify the technique list before seeing any
indicator values, and freeze that classification. Without a survey, this is the closest
thing left to an independent human judgment, and it is worth protecting from contamination
by the analyst's expectations.

## V2 — Subfield-gradient face validity
The D1 prediction in [`04`](04-study-design.md) §3: a gradient from estimation and control
(low deficit) through planning and perception to manipulation, soft robotics, and
fabrication (high). Register the predicted ordering *in advance*, in writing, before
computing it. This converts a soft plausibility check into a real, pre-committed test, and
it is the primary face-validity gate on the whole battery.

## V3 — Codification-event dose-response *(primary anchor)*
Predictive validity through an event study: indicators should drop discontinuously at
codification events (S4), and — the part that makes it a strong test rather than a weak
one — **drop in proportion to how much the event codifies**. A full-pipeline release should
move them more than a partial one; a standard platform more than a one-off script; a
maintained library more than an abandoned dump.

Build the dose scale by hand for ~200 curated events, blind to indicator values. A
monotone dose-response across four or five ordered levels is a much harder thing to get by
accident than a single before/after difference, and it is the strongest evidence the design
can now produce that the indicators track codification rather than fashion.

## V4 — Reproduction ground truth
Sparse, but real labels, and independent of everything else:
- **F1 extractions** — papers stating they could not reproduce something, or needed author
  contact. Free, and now disproportionately valuable. Prioritize the extractor.
- **RA-L reproducible-article track** and robotics reproducibility challenges: a vetted
  positive class. *Verify current status of the programme.*
- **Raff (2019)** and successor ML reproducibility datasets — out-of-domain but directly
  comparable. If our indicators predict his reproduction outcomes, that is genuine external
  validity from a source with no relationship to our corpus construction.
- **Course-based reimplementation.** If the project has access to a robotics course,
  assigning reimplementations of sampled papers generates ground truth cheaply and is
  defensible on its own pedagogical terms. **With the survey tabled this is the best
  remaining route to a human-generated construct anchor, and it is worth actively
  pursuing** — it recovers much of what the survey would have given, at lower cost and with
  no IRB burden beyond the usual coursework arrangements.

## V5 — Placebo and negative controls
- **Style placebo.** A measure built from purely stylistic features with no theoretical
  relation to tacitness (sentence length, passive-voice rate, readability, citation
  density). If it reproduces our headline results, the results are about writing, not
  knowledge. Run this against *every* headline claim, not once.
- **Field control.** Apply the battery to venues where the answer is known a priori — a
  theory venue (low: the proof is the artifact) and a wet-lab or materials venue (high). If
  robotics does not land between them in the expected place, stop.
- **Shuffled-lineage control.** Permute the co-authorship graph; the transfer-channel
  effects must vanish.
- **Parser-degradation control.** Compute every syntax-dependent indicator on a sample of
  old papers that also exist in clean digital form, and confirm the measure does not shift
  with source quality. Parse quality almost certainly degrades going back in time, and that
  degradation would masquerade as a genuine forty-year trend — a threat that did not matter
  when the underdetermination channel was model-based and matters a great deal now.

## V6 — Inter-channel convergence *(now load-bearing)*
With no external human anchor, agreement between channels built from **disjoint data
sources** is carrying more weight than it should have to. Treat it as a real validation
line and design for it:
- The transfer channel uses citation and authorship metadata; the repair and
  underdetermination channels use full text; the substitution channel uses artifact records.
  These are genuinely independent measurement systems, and their agreement is informative
  in a way that agreement between two text measures is not.
- Maintain a **dependency matrix** of indicator → data source, and check it before claiming
  any two channels agree. If two "independent" channels both derive from code-release data,
  their agreement is an artifact.
- Report disagreement as a finding, not a failure ([`02`](02-detection-methods.md) §7).

---

## Threats to validity

### T-A — Writing-convention drift *(highest severity for linguistic indicators)*
Forty years of changing page limits, template rules, and review norms. Any linguistic time
trend is confounded with all of them.
**Mitigations:** within-year comparisons as the default; venue-policy fixed effects; a
hand-compiled page-limit and supplementary-policy timeline; the style placebo.
**Rule: no headline claim rests on a linguistic indicator's raw time trend.**
Cross-sectional and event-study designs are safe; long-run linguistic trends are not.

*Note:* the §0 no-LLM constraint removes one version of this threat — we are not measuring a
moving target with a moving instrument — but it introduces the parser-degradation variant in
V5, which is the same problem wearing different clothes. Watch it just as carefully.

### T-B — Openness selection
Covered in [`03`](03-corpus.md) §3.4. Contained by the IEEE API for Layer A, unresolved for
Layer B. Restated here because it remains the most likely source of a wrong *published*
number, and because it is wrong in the flattering direction.

### T-C — Circularity
If tacitness is defined partly by code availability and validated against code availability,
we have proved nothing. **Rule:** every validation must use a data source disjoint from the
indicators being validated. This bites harder now that V3 (a codification-event measure) is
the primary anchor — V3 must be run against indicators that do *not* themselves consume
artifact-release data, which in practice means the repair, underdetermination, and
epistemic-base channels, not S1–S4. Check the dependency matrix before every V3 claim.

### T-D — Survivorship
Published papers are successful projects. Tacit knowledge that killed a project never enters
the corpus, so our deficit is a lower bound, biased toward techniques that worked. Partially
addressable through negative results in ablation sections and workshop papers; mostly a
limitation to state. Note this is one of the places the survey would have helped most, since
practitioners remember the failures the literature does not record.

### T-E — Technique identity instability
Discussed in [`04`](04-study-design.md) §1. Report key results at two granularities.

### T-F — Uncodified ≠ tacit
Cowan, David & Foray's objection: much of what we detect may be knowledge that could have
been written cheaply and simply wasn't. Not a flaw to hide but the study's most interesting
internal distinction. Knowledge codified shortly after we flag it was RTK, and the flag was
correct; knowledge that resists repeated codification attempts is a candidate for STK/MTK.
**Persistence under codification pressure becomes the empirical criterion separating the
kinds** — which turns the objection into a measurement, and does so without needing a survey.

### T-G — Attention and scale confounds
Big, fashionable techniques diffuse fast for reasons unrelated to tacitness. Control for
attention (citations, papers/year, funding) throughout the transfer channel; prefer
within-technique designs.

### T-H — Author disambiguation error
Systematically worse for non-Western names ([`03`](03-corpus.md) §3.5). Report error rates
broken out by name origin, and re-run transfer-channel results on the ORCID-only subset as
a robustness check.

### T-I — Analyst degrees of freedom
The indicator battery is large and the hypothesis set is rich. **Pre-register** the
hypotheses in [`04`](04-study-design.md) §4, the predicted D1 ordering (V2), and the primary
indicator specification before running confirmatory analyses. Seal a random 20% of
techniques until the specification is fixed. Report all pre-registered results including
nulls. This matters more without a survey, not less: with fewer external anchors, the
discipline has to come from pre-commitment.

### T-J — Underpowered cycle detection *(new, from the revised H5)*
H5 now predicts an oscillation, and measurement error attenuates cycle detection — it biases
toward finding *no* oscillation. Two consequences: run a power analysis before estimating
H5a/H5b and report it; and report a null as **underpowered** rather than as evidence against
oscillation unless the power analysis licenses the stronger reading. The H5d lifecycle
design exists partly to mitigate this, since it pools across techniques instead of demanding
long individual series.

---

## Stopping conditions

Stated in advance so the decision is not made under sunk-cost pressure.

- **V2 fails** — the subfield gradient does not appear in the pre-registered direction. The
  instrument is not measuring tacitness. Stop and rebuild.
- **V3 shows no dose-response** — indicators do not move at codification events, or move
  without regard to how much the event codifies. This is now the load-bearing validation;
  if it fails, the construct is not established and the findings paper should not be
  written. The methods paper could still be published as a negative result, but only with
  V1, V4 and V5 strong enough to make the negative credible.
- **V5 style placebo reproduces the headline results** — we are measuring prose. Drop the
  linguistic channels and continue on transfer, failure, and substitution alone.
- **Full text unobtainable** — pivot to a Layer A/A′ study on the transfer and substitution
  channels plus B2. Narrower, still novel, still Mokyrian, and per
  [`03`](03-corpus.md) §3.2 this is a real study rather than a consolation prize.
