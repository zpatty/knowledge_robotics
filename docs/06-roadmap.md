# 06 — Roadmap

Sequenced so that the highest-risk assumptions are tested first and cheaply, and so that no
phase depends on an access route that may not arrive.

## Phase 0 — Feasibility (weeks 1–4)

Purpose: kill the project early if it deserves to be killed.

1. **Start the IEEE TDM licence conversation immediately.** Longest lead item; everything
   in Layer B waits on it. Do this in week 1 regardless of anything else.
2. **Fix the environment's network policy** ([`03`](03-corpus.md) §3.6) or move corpus
   assembly elsewhere.
3. **Assemble Layer A** — full ICRA + IROS metadata and citation graph from OpenAlex/Crossref/
   DBLP. Verify the actual paper counts and replace every estimate in these documents.
4. **Measure arXiv/OA overlap by year.** This number determines the shape of the whole
   project. Report it before designing anything further.
5. **ESP-B pilot on 200 papers** — stratified by decade and subfield. Does divergence vary
   sensibly? Run the ablation and paraphrase checks on 20 of them.
6. **Hand-annotate 300 sentences** for the R1 classes; measure inter-annotator agreement.
   If humans cannot agree on what craft advice is, no classifier will help.

**Gate:** ESP-B pilot shows signal and survives ablation; OA overlap ≥ ~25% in the modern
era; R1 annotation κ ≥ 0.6. If all three fail, restructure per
[`05`](05-validation-and-threats.md) §4.

## Phase 1 — Minimum viable study (weeks 5–10)

A self-contained, publishable result that uses **no full text at all**, so it is immune to
the access risk. Deliberately front-loaded.

- Seed technique registry: **20 techniques**, hand-curated, spanning the algorithm↔hardware
  axis, each with dated origin papers and a dated codification event.
- Actor graph and lineage distances from Layer A.
- Compute **T1 (independent adoption latency)** and **T2 (lineage ratio)** for all 20.
- Compute **S1/S2/S4** from metadata and artifact links.
- Run the **H2 difference-in-differences** on this small set, with pre-trend tests.
- Run **V1 (known cases)** and **V5 (shuffled-lineage placebo)**.

**Output:** a workshop paper or preprint — *"Techniques that travel with people: measuring
tacit knowledge transfer in robotics from bibliometric traces alone."* Also the thing to
show collaborators and funders.

## Phase 2 — Full-text instruments (weeks 8–20, overlapping)

Begins as soon as any full text arrives; scoped to whatever subset exists.

- Ingest and parse (GROBID + LaTeX); build the section-structured store.
- Train R1 classifier; compute R1–R4.
- Compute B1–B4 and F1–F4.
- Run ESP-A/B at scale; ESP-C/D on the code-linked subset.
- Build abstract-only reduced-form variants and calibrate ([`03`](03-corpus.md) §3.3).
- Fit the factor model ([`02`](02-detection-methods.md) §7). **Report the factor structure
  as a finding** — how many dimensions the deficit actually has is a real question and we
  should not assume the answer.

## Phase 3 — Validation (weeks 14–26, overlapping)

- **Launch the expert survey early** — IRB lead time is real, and V2 gates the credibility
  of everything. Do not let this slip to the end.
- V3 reproduction ground truth; V4 codification-event dose-response; full V5 battery.
- The insider/outsider CTK sub-study.
- **Pre-register** [`04`](04-study-design.md) §4 before the confirmatory analyses; seal the
  20% hold-out.

## Phase 4 — Analysis (weeks 24–40)

- Descriptives D1–D5.
- Confirmatory H1–H7. The spine is **H2** (codification → independent diffusion), **H4**
  (conserved frontier), **H5** (Ω↔λ direction of travel).
- The six case studies, read qualitatively, in parallel with the quantitative work rather
  than after it — they are how we catch the instrument lying.

## Phase 5 — Write-up and release (weeks 36–52)

Two papers and the dataset release per [`04`](04-study-design.md) §6.

---

## Fast path

If only six weeks exist: **Phase 0 items 3–5, then Phase 1.** That yields the transfer-channel
result on 20 techniques plus the ESP-B pilot. It is a real contribution on its own and it
de-risks everything else.

## Team and skills

Roughly: one person on corpus/infrastructure (bibliometric plumbing, disambiguation, parsing);
one on NLP/LLM instruments (R, B, ESP); one on causal inference (panel methods, event studies,
survival analysis); one domain roboticist for the registry, case studies, and expert survey
design. The domain roboticist is not optional — technique identity and subfield judgment
cannot be outsourced to a classifier, and getting them wrong invalidates the panel.

## Open decisions

These are yours to make; the plan is written to accommodate either branch of each.

1. **Full-text route.** Pursue the IEEE TDM licence (slower, complete, removes the openness
   bias), or accept an arXiv/OA-only corpus (fast, biased, requires the correction machinery
   in [`03`](03-corpus.md) §3.3)? My recommendation: start the licence in week 1 *and* build
   on OA in the meantime — they are not exclusive, and the plan is sequenced so nothing
   blocks on the licence.
2. **Emphasis.** A methods paper (novel instrument for measuring codification deficit,
   audience = scientometrics/STS) or a findings paper (Mokyr's framework tested on 40 years
   of robotics, audience = economics of innovation)? The plan produces both, but which one
   leads determines where validation effort versus modeling effort goes. My recommendation:
   lead with methods, because the instrument is the genuinely new thing and the findings are
   only as credible as it is.
3. **Scope of the expert survey.** A light validation instrument (~30 people, ~50 items) or
   a substantive component with its own findings (the insider/outsider CTK study)? The
   latter costs more and is, I think, the better paper.
4. **Whether to include journals** (T-RO, RA-L, IJRR) in the main panel. Including them
   captures technique maturation but breaks the clean venue-normalization story. Suggest:
   conference-only for the main panel, journals as a secondary analysis.
