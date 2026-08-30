# 06 — Roadmap

Sequenced so that the highest-risk assumptions are tested first and cheaply, and so that no
phase depends on an access route that may not arrive.

**Current scope constraints** (set by the PI, and reflected throughout these documents):
no expert survey; no LLM-based text assessment. Both are recorded with their re-entry
conditions in [`appendix-deferred.md`](appendix-deferred.md). IEEE API access is available;
full-text/TDM rights are a separate matter and not yet in hand.

## Phase 0 — Feasibility (weeks 1–4)

Purpose: kill the project early if it deserves to be killed.

1. **Start the IEEE TDM licence conversation immediately.** Longest lead item; all of
   Layer B waits on it, and it is *separate* from API access. Week 1, regardless of
   anything else.
2. **Fix the environment's network policy** ([`03`](03-corpus.md) §3.7) or move corpus
   assembly elsewhere.
3. **Harvest Layer A from the IEEE API.** Verify actual ICRA/IROS paper counts and replace
   every estimate in these documents. Then the critical measurement: **the
   abstract-coverage-by-year curve**, plus whether affiliations and supplementary-material
   flags are returned for the early years. The "measure a complete corpus" argument in
   [`03`](03-corpus.md) §3.3 stands or falls on this, and everything downstream is scoped
   by it.
4. **Assemble Layer A′** — the reference graph, from OpenAlex/Crossref. Check merge quality
   against the IEEE record.
5. **Measure arXiv/OA full-text overlap by year.** Determines how much of Phase 2 is
   possible at all.
6. **Check the H5 sample constraint.** How many techniques will plausibly have ≥ 15 years
   of usable history? If the answer is small, H5a/H5b are underpowered from the start and
   H5d (the lifecycle design) becomes the primary test — better to know this in week 3 than
   in month 8.
7. **Hand-annotate 300 sentences** for the R1 classes against a written guideline; measure
   inter-annotator agreement.
8. **Pre-register the D1 subfield ordering** (V2) before computing anything.

**Gate:** abstract coverage adequate back to at least ~1995; OA full-text overlap ≥ ~25% in
the modern era; R1 annotation κ ≥ 0.6. Failure on the first reshapes the study rather than
ending it (pre-1995 becomes metadata-only); failure on all three means restructuring per
[`05`](05-validation-and-threats.md), *Stopping conditions*.

## Phase 1 — Minimum viable study (weeks 5–10)

Self-contained and publishable, using **no full text at all** — immune to the TDM risk and
untouched by both scope constraints. Deliberately front-loaded.

- Seed technique registry: **20 techniques**, hand-curated, spanning the algorithm↔hardware
  axis, each with dated origin papers and dated codification events.
- Actor graph and lineage distances from Layers A/A′.
- **T1** (independent adoption latency) and **T2** (lineage ratio) for all 20.
- **S1/S2/S4** from metadata and artifact links — including, if the IEEE flags exist, the
  forty-year video-attachment series, which is a publishable descriptive result by itself.
- **B2** (reference reach and depth) — the one epistemic-base indicator computable from
  metadata, and therefore the one that carries the first pass at H5.
- **H2 difference-in-differences** with pre-trend tests.
- **H5d** on this small set: the technique-age lifecycle profile of B2 against the
  transfer-channel deficit. Underpowered at n=20, but it establishes whether the cyclical
  signature is visible at all before committing to the full panel.
- Validation: **V1** (blind known-case), **V5** (shuffled-lineage and style placebos).

**Output:** a preprint — *"Techniques that travel with people: measuring tacit knowledge
transfer in robotics from bibliometric traces alone."*

## Phase 2 — Text instruments (weeks 8–20, overlapping)

Two tracks, because they have different access dependencies:

**2a — Abstract-level, corpus-wide** (needs only Layer A; start immediately):
- Subfield assignment from IEEE index terms + embeddings.
- B3, and reduced-form abstract variants of R1 and U2.
- Calibrate reduced forms against full-text counterparts once 2b exists.

**2b — Full-text** (scoped to whatever subset exists):
- Ingest and parse (GROBID + LaTeX, preferring LaTeX); section-structured store.
- **F1 first** — cheap, and the only near-ground-truth in the design.
- **U1 checklist authoring** — the domain roboticist's highest-value task, and the critical
  path for the underdetermination channel. ~15 families × 12–25 items.
- U2 (delegation density and chain depth), U3 (hedge attachment).
- R1 classifier training; R1–R4.
- B1, B4; F2–F4.
- Fit the factor model ([`02`](02-detection-methods.md) §7). **Report the factor structure
  as a finding** — how many dimensions the deficit has is a real question.

## Phase 3 — Validation (weeks 14–26, overlapping)

- **V3 codification-event dose-response** — now the primary anchor. Hand-build the ~200-event
  dose scale *blind* to indicator values. This is the most important validation work in the
  project and should not be squeezed.
- V4 reproduction ground truth; pursue the course-based reimplementation route actively,
  since it is the best remaining human anchor.
- Full V5 battery, including the parser-degradation control.
- V6 dependency matrix and inter-channel convergence.
- **Pre-register** [`04`](04-study-design.md) §4; seal the 20% hold-out.

## Phase 4 — Analysis (weeks 24–40)

- Descriptives D1–D5.
- Confirmatory H1–H7. The spine is **H2** (codification → independent diffusion) and the
  joint **H4 + H5** (conserved frontier / oscillating Ω↔λ relationship), estimated and
  reported together as one mechanism.
- Power analysis for H5a/H5b before estimation (T-J).
- The six case studies, read qualitatively **in parallel** with the quantitative work rather
  than after it — they are how we catch the instrument lying, and with the survey tabled
  they are also the main source of interpretive grounding.

## Phase 5 — Write-up and release (weeks 36–52)

Two papers and the dataset release per [`04`](04-study-design.md) §6.

---

## Fast path

If only six weeks exist: **Phase 0 items 3–6, then Phase 1.** Yields the transfer-channel
result on 20 techniques plus a first look at the H5 lifecycle signature. A real contribution
on its own, and it de-risks everything else.

## Team and skills

One person on corpus infrastructure (API harvesting, disambiguation, parsing); one on
computational text analysis (lexicons, annotation, supervised classifiers, dependency
parsing — *not* an LLM-tooling role, given the §0 constraint); one on causal inference
(panel methods, event studies, survival analysis, and now spectral/regime-switching methods
for H5); one domain roboticist. The roboticist is not optional and their load has *risen*
under the current constraints — technique identity, the U1 checklists, the blind V1 and V3
classifications, and the case studies all require them, and none can be outsourced to a
classifier.

## Open decisions

1. **Full-text route.** TDM licence (slower, complete) versus arXiv/OA-only (fast, biased).
   *Recommendation:* start the licence in week 1 *and* build on OA meanwhile — not exclusive,
   and nothing in the plan blocks on it.
2. **Emphasis.** Methods paper or findings paper first? *Recommendation:* this has shifted.
   With the model-based probe deferred, the methods contribution is less novel — U1/U2/U3
   are careful rather than new — while the revised H5 (an oscillating Ω/λ relationship,
   tested spectrally over forty years) is a genuinely original empirical claim. **Lead with
   the findings paper**, and let the methods stand as its instrument section plus a
   dataset release.
3. **Journals in the main panel?** T-RO/RA-L/IJRR capture technique maturation but break the
   venue-normalization story. *Suggestion:* conference-only main panel, journals secondary.
4. **Course-based reimplementation** ([`05`](05-validation-and-threats.md) §V4) — worth
   arranging? It is the best remaining human-generated construct anchor and the cheapest
   partial substitute for the survey.
