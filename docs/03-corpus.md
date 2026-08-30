# 03 — Corpus construction

## 3.1 Scope

**Primary corpus:** all papers published in IEEE ICRA (International Conference on Robotics
and Automation, from 1984) and IEEE/RSJ IROS (Intelligent Robots and Systems, from 1988) to
the present. Rough order of magnitude 55–75k papers; recent years run ~1,200–1,500 accepted
papers each. *These figures are estimates and must be verified against the actual index
before any of them appear in writing — see §3.6.*

Why these two venues: they are the field's continuous, high-volume core; together they span
four decades of robotics with stable identity; and they cover the whole algorithm↔hardware
range in one place, which is precisely the axis along which tacitness should vary.

**Secondary corpora**, for specific jobs — not for the main panel:
- **RA-L / T-RO / IJRR** — journal versions; needed to follow techniques that mature out of
  the conference track, and RA-L carries the reproducible-article track (a validation set).
- **arXiv cs.RO** — the preprint layer; essential because appendices and implementation
  details are frequently longer in the preprint than in the camera-ready. *The
  preprint↔camera-ready diff is itself a measurement instrument*: what gets cut under a page
  limit is a direct observation of enforced decodification. Worth its own short paper.
- **CoRL, RSS, workshop proceedings** — for subfield coverage and codification events.
- **Non-robotics control group** — a venue with comparable size and different embodiment
  (e.g. a theory venue and a systems venue) to establish that our measures separate fields
  in the expected direction.

## 3.2 The three data layers

| Layer | Content | Sources | Coverage expected |
|---|---|---|---|
| **A — Metadata** | Titles, authors, affiliations, years, references, citation graph | OpenAlex (free, complete-ish, has references), Crossref, DBLP, Semantic Scholar API | ~complete |
| **B — Full text** | Sections, method text, tables, acknowledgments, captions | arXiv, S2ORC, Unpaywall/OA copies, IEEE TDM licence | **partial — the binding constraint** |
| **C — Artifacts** | Code repos, datasets, videos, benchmarks, platforms | GitHub API, dataset DOIs, paper-with-code archives, IEEE supplementary material index | partial, biased recent |

**Layer A is enough to run the entire transfer channel** (T1–T4) and most of the
substitution channel. This matters enormously for project risk: if full-text access falls
through, the study's strongest indicator family survives intact. Sequence the work
accordingly — see [`06`](06-roadmap.md).

## 3.3 Full-text access: the real constraint

IEEE Xplore does not permit bulk scraping, and the terms are enforced. Routes, in order of
preference:

1. **IEEE Text and Data Mining licence** through the institution (Case Western Reserve).
   This is the correct route and should be initiated in week 1 — it is the longest-lead item
   in the project and everything in Layer B waits on it. Expect an institutional-library
   conversation plus an IEEE agreement; budget 4–12 weeks and start now.
2. **arXiv overlap.** Large and growing for robotics, near-total for recent learning-heavy
   work, essentially absent before ~2010. Free, clean LaTeX source (which is *better* than
   PDF — real section structure, real math, real tables).
3. **S2ORC / Semantic Scholar open full text** for the openly licensed remainder.
4. **Unpaywall / author-posted copies** for the long tail, with per-item licence checking.

Parsing: GROBID for PDFs; direct LaTeX parsing where arXiv source is available (prefer it —
substantially higher fidelity for tables, numbers, and section boundaries, which is exactly
what R3 and F2 depend on).

### The openness-selection problem — the most serious threat in the corpus design

Full-text availability is **not** missing-at-random with respect to our outcome. Authors who
post preprints and open-license their work are systematically the same authors who release
code, use standard platforms, and write more completely. If we measure codification deficit
only on the open subset, we measure it on the least tacit part of the field, and any time
trend will be contaminated by the growth of openness itself.

Mitigations, all of which should be in the paper:

- **Abstract-only indicator variants.** Build reduced-form versions of R1, B3, and ESP-A
  that run on abstracts alone, which are available for the entire corpus. Calibrate them
  against full-text versions on the open subset, then apply corpus-wide. Report both.
- **Inverse-probability weighting.** Model P(full text available | year, subfield, venue,
  affiliation region, author openness history) and weight.
- **Leaning on Layer A.** The transfer channel needs no full text at all. Where the two
  channels agree on the open subset, the transfer channel can carry the full-corpus claim.
- **Never report an uncorrected corpus-wide time trend from full-text-only indicators.**
  This is the single easiest way to publish a wrong headline number, and it would be a
  wrong number in the *flattering* direction (the field looking like it codified more than
  it did).

## 3.4 Normalization and disambiguation

- **Author disambiguation.** ORCID where available; otherwise an ensemble of name +
  affiliation + co-author + topic signals. Hand-check a stratified sample of 500 and report
  precision/recall, broken out by name origin — disambiguation error is systematically worse
  for East Asian names, and since a large share of robotics output is from East Asian
  institutions, an unreported error rate here would silently distort the entire transfer
  channel.
- **Affiliation → institution → country**, with histories, using ROR.
- **Subfield assignment.** Do not use venue-supplied session tracks alone (they change every
  year). Build a stable topic layer: embed abstracts, cluster, hand-label ~40 subfields,
  then assign with a classifier. Validate against a hand-labeled sample and against the
  IEEE RAS technical-committee taxonomy.
- **Venue year normalization.** Page limits, template changes, and supplementary-material
  policy changes are structural breaks. Compile a venue-policy timeline by hand and include
  it as fixed effects everywhere.

## 3.5 Storage and reproducibility

Plain and boring on purpose: Parquet for the panel, DuckDB for analysis, a content-addressed
raw store for retrieved documents, and a manifest recording retrieval date and licence for
every item. Every derived table regenerable from raw by a single pipeline command. Full text
under restrictive licences stays local and is never redistributed; derived aggregate
features and indicator values are what get released.

There is a pleasing reflexivity here worth naming in the paper: **the study's own output —
a public technique registry with dated codification events and indicator values — is itself
a codification artifact**, and one that lowers access costs to exactly the kind of knowledge
Mokyr says matters. Build it to be used.

## 3.6 Immediate blocker in this environment

This session's network policy denies outbound HTTPS to everything but package registries, so
the OpenAlex/arXiv feasibility probes that would firm up the counts in §3.1 could not be
run. Before Phase 0 work begins, either allowlist `api.openalex.org`, `api.crossref.org`,
`export.arxiv.org`, `api.semanticscholar.org`, `api.github.com`, and `api.ror.org` in the
environment's network configuration, or run corpus assembly outside this environment. Until
then, treat every corpus count in these documents as an unverified estimate.
