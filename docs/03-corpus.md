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
| **A — Venue + abstract** | ICRA/IROS membership, titles, authors, affiliations, abstracts, index terms, supplementary flags | **OpenAlex** (bulk), **IEEE Xplore API** (authority + gap-fill — see §3.3), DBLP | target: complete and unbiased |
| **A′ — Citation graph** | References, citation contexts | OpenAlex (primary — has the reference edges), Crossref, Semantic Scholar | ~complete |
| **B — Full text** | Sections, method text, tables, acknowledgments, captions | arXiv, S2ORC, OA copies, IEEE TDM licence | **partial — the binding constraint** |
| **C — Artifacts** | Code repos, datasets, videos, benchmarks, platforms | GitHub API, dataset DOIs, IEEE supplementary index | partial, biased recent |

**Layers A and A′ are enough to run the entire transfer channel** (T1–T4), most of the
substitution channel, and B2 — which means a first pass at H2, H3, H4 and H5 needs no full
text at all. This matters enormously for project risk. Sequence the work accordingly; see
[`06`](06-roadmap.md).

## 3.3 What the IEEE API does and does not fix

### The rate limit reverses the intended roles

The granted IEEE key allows **200 calls per day** (10/sec). That single number
overturns the source architecture sketched above, and it is worth being explicit about
the arithmetic:

- At a per-call ceiling of ~200 records (*assumed, must be verified — see the probe*),
  the theoretical maximum is 40,000 records/day. A ~65k-paper corpus is therefore a
  **two-day pull in the perfect case**, and realistically three to five once probing,
  retries, and per-venue-year partitioning are accounted for.
- More to the point: **one wasted call is 0.5% of a day's allowance, and one buggy loop
  is the whole day.** Any design where IEEE is the bulk workhorse is a design where a
  single mistake costs a week.

So the roles invert from the first draft:

| | Role | Why |
|---|---|---|
| **OpenAlex** | **Bulk workhorse** | Generous budget with the free key (10× keyless), cursor paging past 10k, references included, and abstracts via `abstract_inverted_index`. A full ~65k-work pull is ~650 calls at `per-page=100` — a comfortable afternoon, not a multi-day operation. |
| **IEEE Xplore** | **Authority and gap-filler** | Spend the 200/day only on what nothing else has. |

**What the IEEE budget should actually be spent on**, in priority order:

1. **Completeness audit.** One call per venue-year returns `total_records`. Eighty calls
   covers both venues across forty years and gives an authoritative count to check
   OpenAlex against. This is the highest-value use of the budget by a wide margin — it
   converts "we think we have the corpus" into "we know what we are missing, by year."
2. **Abstract gap-filling.** Harvest abstracts from OpenAlex first, measure the gap,
   then spend IEEE calls only on the years where OpenAlex is thin. `coverage_report.py`
   computes the exact call cost of closing the gap before any of it is spent.
3. **IEEE controlled index terms**, for the subfield layer.
4. **Supplementary-material / multimedia flags** — *if* the API exposes them (unverified).

### What it fixes

- **Authoritative venue membership.** OpenAlex source records for long-running
  conference series are messy — renamed proceedings, split records, per-year records,
  missing years. Picking one source ID at random silently truncates the corpus, and an
  IEEE venue-year count catches exactly that class of error.
- **Complete, unbiased abstracts.** The openness-selection problem below is a problem
  about *full text*. If abstracts are complete corpus-wide, every abstract-computable
  indicator — B3, the reduced forms of R1 and U2, subfield assignment — runs
  **corpus-wide with no selection bias at all**. That is a move from "reweight a biased
  sample" to "measure a complete one." Whether we get there is now a *joint* OpenAlex +
  IEEE question rather than an IEEE one.
- **Controlled index terms** spanning the full period — better than clustering for
  subfield assignment, and stable across years in a way embeddings are not.
- **Supplementary-material flags**, if exposed: video-attachment presence for the whole
  corpus across four decades, unbiased. Video is the field's main non-propositional
  carrier (S1), and a complete forty-year series on it is a publishable descriptive
  result on its own.

### What it does not fix

- **Full text.** The standard Xplore API returns metadata and abstracts, not article
  full text — that is the separate TDM agreement, and **having an API key does not imply
  having TDM rights**. R1–R4, U1, U3, F2–F4 and B1/B4 stay restricted to the open
  subset, with the machinery in §3.4 applying to those indicators specifically.
- **The reference graph.** OpenAlex and Crossref remain primary for Layer A′.

### Probe before harvest

Field availability across four decades is unverified, and a 2023 record tells us nothing
about a 1986 one. `scripts/probe_ieee.py` spends ~8 calls to establish: whether the key
is active at all (the granted key's status reads *waiting*, which may mean not yet
approved); and which fields — abstract, affiliations, index terms, multimedia — are
actually returned per era. **Those eight calls determine the entire harvest design and
are worth more than eight hundred calls of bulk data.** Run it with `--dry-run` first.

Two questions the probe was originally meant to answer are already settled, from the
2020 code on the `master` branch ([`07`](07-harvest-operations.md), *What the 2020 code
already settles*): the `max_records` ceiling is genuinely 200 and larger requests are
*clamped rather than rejected*, and deep `start_record` paging works. The same source
corrects the venue title strings — the working forms carry no `IEEE` prefix.

**The abstract-coverage-by-year curve remains the single most important number to
establish in week 1.** If coverage thins before ~1995, the pre-1995 analysis is
metadata-only and should be planned that way from the start rather than discovered in
month three.

## 3.4 Full-text access and the openness-selection problem

Routes to full text, in order of preference:

1. **IEEE Text and Data Mining licence** through the institution (Case Western Reserve).
   The correct route, and the longest-lead item in the project — start it in week 1 and
   expect 4–12 weeks. Note it is a *separate* agreement from API access; having the API
   does not imply having TDM rights.
2. **arXiv overlap.** Large and growing for robotics, near-total for recent learning-heavy
   work, essentially absent before ~2010. Free, and the LaTeX source is *better* than PDF —
   real section structure, real math, real tables, which is exactly what R3, U1 and F2
   depend on.
3. **S2ORC / Semantic Scholar** open full text for the openly licensed remainder.
4. **Unpaywall / author-posted copies** for the long tail, with per-item licence checking.

Parsing: GROBID for PDFs; direct LaTeX parsing where arXiv source exists (strongly prefer
it). Check parser behaviour on 1980s two-column scans early — dependency-parse quality
almost certainly degrades going backward in time, and that degradation would masquerade as
a genuine time trend in U3 and in R1's syntax-dependent classes.

### The openness-selection problem

Full-text availability is **not** missing-at-random with respect to our outcome. Authors
who post preprints and open-license their work are systematically the same authors who
release code, use standard platforms, and write more completely. Measure the deficit only
on the open subset and you measure it on the least tacit part of the field, and any time
trend is contaminated by the growth of openness itself.

With the IEEE API in hand this threat is **contained rather than solved**, and the
containment strategy is now clean:

- **Tier the claims by data layer.** Corpus-wide claims come from Layers A/A′ only
  (transfer channel, substitution channel, B2, B3, abstract-level indicators). Full-text
  indicators support *within-open-subset* comparisons and cross-sectional analysis, never a
  corpus-wide time trend. Say which tier each result belongs to, in the results tables.
- **Calibrate the reduced forms.** Build abstract-only variants of the full-text
  indicators, calibrate them against their full-text counterparts on the open subset, then
  apply corpus-wide. Report both, and report the calibration error honestly.
- **Inverse-probability weighting** for the full-text indicators: model P(full text
  available | year, subfield, venue, region, author openness history) and weight.
- **Rule: never report an uncorrected corpus-wide time trend from a full-text-only
  indicator.** This is the easiest way to publish a wrong headline number, and it would be
  wrong in the *flattering* direction — the field looking like it codified more than it did.

## 3.5 Normalization and disambiguation

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

## 3.6 Storage and reproducibility

Plain and boring on purpose: Parquet for the panel, DuckDB for analysis, a content-addressed
raw store for retrieved documents, and a manifest recording retrieval date and licence for
every item. Every derived table regenerable from raw by a single pipeline command. Full text
under restrictive licences stays local and is never redistributed; derived aggregate
features and indicator values are what get released.

There is a pleasing reflexivity here worth naming in the paper: **the study's own output —
a public technique registry with dated codification events and indicator values — is itself
a codification artifact**, and one that lowers access costs to exactly the kind of knowledge
Mokyr says matters. Build it to be used.

## 3.7 Immediate blocker in this environment

This session's network policy denies outbound HTTPS to everything but package registries, so
the OpenAlex/arXiv feasibility probes that would firm up the counts in §3.1 could not be
run. Before Phase 0 work begins, either allowlist `ieeexploreapi.ieee.org`, `api.openalex.org`,
`api.crossref.org`, `export.arxiv.org`, `api.semanticscholar.org`, `api.github.com`, and
`api.ror.org` in the
environment's network configuration, or run corpus assembly outside this environment. Until
then, treat every corpus count in these documents as an unverified estimate.
