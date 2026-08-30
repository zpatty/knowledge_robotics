# CLAUDE.md

Orientation for Claude Code sessions in this repository.

## What this project is

**"The Unwritten Half"** — a study, after Joel Mokyr's *The Gifts of Athena*, of the
relationship between what roboticists know how to do (λ-knowledge, prescriptive) and
what they can write down (Ω-knowledge, propositional), measured across forty years of
IEEE ICRA and IROS proceedings (1984–present, ~55–75k papers).

The measured construct is the **codification deficit** `D(λ,t)` — what a practitioner
must know to execute a technique, minus what is recoverable from the public record.
It is detected through six channels of "negative space" traces (repair, transfer,
failure, substitution, underdetermination, epistemic base), each with computable
indicators (`R1`–`R4`, `T1`–`T4`, `F1`–`F4`, `S1`–`S4`, `U1`–`U3`, `B1`–`B4`).

**Status: Phase 0 done, and it went badly for the plan's source architecture.**
The design (channels, indicators, hypotheses, validation) is untouched; where Layers
A/A′ come from is not. **Read `docs/08-phase0-findings.md` before touching anything
corpus-related** — it supersedes `03` §3.3. The one-line version: IEEE is 403ing
regardless of key, OpenAlex has <10% of the venue linkage and is now metered, and
Crossref carries a usable corpus from ~2007 with ~99% reference edges. A Crossref
harvest of 2006–2025 is the working corpus.

## Branch topology — read this before assuming what is in the tree

The repository carries three unrelated lines of work. This is the single most
confusing thing about it:

| Branch | Contents |
|---|---|
| `main` | Effectively empty — a one-line `README.md`. The GitHub default branch. |
| `master` | **Zach's original 2020 work**, unrelated to `main`'s history: MALLET/LDA topic modeling over arXiv `cs.RO` abstracts, plus an IEEE Xplore scraper for ICRA/IROS. See below. |
| `claude/tacit-knowledge-robotics-vrcopx` | The research plan and harvest tooling. **The real project.** This branch is its parent. |

`main` is not a superset of anything. A fresh clone lands on a nearly empty tree and
`git branch -a` will not show the other branches unless they are fetched
(`git fetch origin master claude/tacit-knowledge-robotics-vrcopx`).

### The prior work on `master` is not referenced by the plan

`master` contains `robotics_knowledge.ipynb`, `robotics_knowledge_arxiv.py`, and
`xploreapi.py` — an earlier attack on the *same* Ω/λ question by MALLET/LDA topic
modeling, plus IEEE's own distributed Xplore sample client. Its commit log records that
it "can now scrape IEEE Xplore for all ICRA and IROS papers."

**The pipeline itself is superseded and not worth reviving** (the PI's own assessment),
but it retired three of the plan's unverified assumptions about the IEEE API — the
`max_records` ceiling, deep paging, and the venue title strings. Those findings are
written up in `docs/07-harvest-operations.md`, *What the 2020 code on `master` already
settles*, and are already folded into `src/tacit/ieee.py`. Do not re-derive them.

⚠️ **`robotics_knowledge.ipynb` on `master` has an IEEE API key hardcoded in three
cells, in a public repo.** Treat it as compromised and revoke it; it is unrelated to
the current key. Details in `docs/07`.

## Document map

Read in this order; every doc cross-references the others by ID.

| Doc | What it settles |
|---|---|
| `docs/01-framework.md` | Mokyr's Ω/λ; Collins' RTK/STK/CTK plus a fourth kind (MTK, material); the codification deficit; epistemic-base width `B(λ,t)`. |
| `docs/02-detection-methods.md` | **Core contribution.** Six channels, ~20 indicators, each with computation, cost, and confounds. |
| `docs/03-corpus.md` | ICRA/IROS corpus; four data layers; full-text access; the openness-selection problem. |
| `docs/04-study-design.md` | Technique registry as unit of analysis; data model; D1–D5; H1–H7. |
| `docs/05-validation-and-threats.md` | V0–V6, threats T-A–T-J, and explicit stopping conditions. |
| `docs/06-roadmap.md` | Phases 0–5, the 6-week fast path, open decisions. |
| `docs/07-harvest-operations.md` | Operational notes for `src/tacit/` and `scripts/`. |
| `docs/08-phase0-findings.md` | **What the live APIs actually returned.** Supersedes `03` §3.3. Read before any harvest work. |
| `docs/appendix-deferred.md` | The expert survey and the model-in-the-loop probe, with re-entry conditions. |

## Two hard scope constraints, set by the PI

These are binding design decisions, not preferences. Both are easy to violate by
reflex, so check any proposed change against them.

1. **No LLM-based text assessment.** (`02` §0.) Every indicator must be computable
   with lexicons, rule systems, dependency parsing, structural document parsing, or
   supervised classifiers trained on published annotation guidelines. No indicator may
   depend on a language model's judgment of a text. The reasons are instrument
   constancy across forty years, training-data contamination correlated with the time
   axis, and auditability. **Do not propose an LLM classifier as a shortcut for R1, U1,
   U2, B1, or citation-intent labeling** — the deferred version and its four re-entry
   conditions are in `appendix-deferred.md` §B.
2. **No expert elicitation survey.** (`appendix-deferred.md` §A.) This removed the
   design's construct-validity anchor. Consequences that must be respected in any
   analysis or writing: all claims are comparative — rankings, gradients, event-study
   deltas, phase relationships — and **no claim of the form "robotics is X% tacit"**
   may appear. Collective tacit knowledge is out of scope.

## Where the corpus actually comes from

The plan's OpenAlex-bulk / IEEE-authority split (`03` §3.3) did not survive contact
with the live APIs. Current state, per `docs/08`:

| Source | State | Role now |
|---|---|---|
| **Crossref** | free, unmetered, no key | **The working corpus.** ~2007–present, ~99% reference edges. Affiliations only from 2022. |
| **OpenAlex** | metered (1,000 credits/day free); supplied key is rejected, keyless polite pool works | Under 10% venue linkage — IROS resolves to 3 years. Not usable until F3 in `08` is settled. |
| **IEEE Xplore** | `ERR_403_DEVELOPER_INACTIVE` for *any* key, including none | Blocked. Only route to 1984–2006 and to abstracts. |

`src/tacit/crossref.py` documents three silent Crossref traps — read it before
writing a query. The worst: **cursor paging discards relevance ordering**, so paging
a fuzzy query walks arbitrary records and silently returns almost nothing.

## API budget discipline

Still the code's central concern, even though the binding source changed.
IEEE allows **200 calls/day** — one wasted call is 0.5% of a day. OpenAlex meters
**credits, not calls**, scaling with page size (`credits_for(per_page)`); budgeting by
call count under-states a paged harvest ~5×. Crossref is unmetered and its guard is
only a runaway catch.

Four rules are enforced in code, not by discipline:

- `src/tacit/cache.py` — content-addressed on the request URL with credentials
  scrubbed, so a rotated key does not invalidate the cache. Cache-first is the budget
  policy.
- `src/tacit/budget.py` — **reserve-then-call**; the reservation is spent before the
  request goes out, because the API charges for a call whose response fails to parse.
  Persists to `data/budget_ieee.json`; refuses rather than exceeds.
- Every client takes `dry_run=True`. **Always run `--dry-run` first** and read the
  printed URLs.

- A long `Retry-After` raises `QuotaExhausted` rather than sleeping. OpenAlex answers
  a spent quota with `Retry-After: 79803` — 22 hours — and the retry loop used to
  sleep on it.

`IEEEXplore(reserve=N)` holds back N calls so an automated harvest cannot eat the
allowance an interactive probe needs.

### Order of operations

```bash
# The working path today — Crossref, no key, no quota:
python3 scripts/survey_crossref.py --years 2006-2025    # coverage by year
python3 scripts/harvest_crossref.py --years 2006-2025   # the corpus; resumable

# Blocked until the IEEE account is sorted (docs/08 F1):
python3 scripts/probe_ieee.py --dry-run && python3 scripts/probe_ieee.py
```

`harvest_crossref.py` is resumable — it reads existing DOIs from `--out` and skips
them, so an interrupted run continues.

**Any OpenAlex source selection stays manual.** Source records for long-running
conference series are split, renamed, and incomplete; picking one programmatically by
best name match silently truncates the corpus, invisibly. That is exactly how F3 was
found.

## Environment

This environment is **already configured** — keys are in the environment and
`api.openalex.org`, `ieeexploreapi.ieee.org`, `api.crossref.org`, `export.arxiv.org`,
`api.semanticscholar.org` and `api.ror.org` are reachable. If a fresh environment is
not, `docs/07` has the setup: network access **Custom**, those domains allowlisted,
**package-manager defaults kept checked**, keys in the environment-variables box, then
start a new session.

**`dblp.org` is NOT allowlisted** and its CONNECT is refused. Worth adding: DBLP is
the cleanest free source of per-venue-year DOI lists, and a DOI list is precisely what
would unlock the pre-2007 corpus that Crossref holds but cannot enumerate (`08` F4).

Do **not** use the environment's *API credentials* feature for these keys — it injects
HTTP headers, but both APIs authenticate by query string, so the credential attaches,
is ignored, and produces 401s that look like a bad key.

A cloud VM is the wrong home for the harvest regardless: the container is reclaimed
after the session, and a 65k-work corpus should not be committed. Prefer a persistent
machine for harvesting and cloud sessions for analysis and code.

## Two standing options, set by the PI

Do not re-litigate these; assume they are available when weighing a route.

- **Paying for OpenAlex is acceptable** (bounded, not open-ended — it sells
  prepaid credits, so a spent balance 403s rather than billing on). A full-corpus
  pull is ~$0.33. Cost is never the reason to reject an OpenAlex route; coverage
  is (`08` F3).
- **IEEE cooperation is probably obtainable** via the PI's networking, given lead
  time. Routes needing IEEE goodwill — API activation, the TDM licence — are worth
  planning for rather than designing around, as long as nothing on the critical
  path blocks waiting for them.

## What is unverified

Nothing in `src/` or `scripts/` has been run against a live API. Treat every corpus
count in the docs as an estimate. Specifically open:

- Whether the current IEEE key is active at all — its granted status reads *waiting*.
- Which fields IEEE returns per era — abstracts, affiliations, index terms, and
  especially supplementary-material flags, which `S1` leans on heavily.
- OpenAlex abstract coverage before ~2000. **This is the Phase 0 gate**: the
  abstract-coverage-by-year curve is the single most important number to establish.
- Whether the venue title strings hold across all forty years. They worked for a bulk
  pull in 2020, but IROS ran as a *Workshop* before 1989; `VENUE_TITLE_VARIANTS` in
  `ieee.py` carries the alternates, and the completeness audit is what catches a
  variant that silently returns zero for an era.

Settled from `master` and no longer open: the 200-record per-call ceiling (and that
larger requests are **clamped, not rejected**), deep `start_record` paging, and the
venue title strings.

## Working notes for Claude

- **Cite the doc IDs.** The documents form a cross-referenced system (`H5d`, `V3`,
  `T-F`, `U2`, `B2`). When changing one, check what references it — the docs were
  revised once already (H5 moved from directional to cyclical) and the ripple touched
  the statistics, the roadmap, and the validation plan.
- **The load-bearing pieces**, if you need to know what not to casually undermine:
  `V3` (codification-event dose-response) is the primary validation anchor now that the
  survey is gone; `B2` (reference reach) is the only epistemic-base indicator computable
  back to 1984 and therefore carries H5; `T1`/`T2` are the strongest indicator family
  and need no full text.
- **Layer A/A′ (metadata) is enough for the whole Phase 1 study.** If full text is
  blocked, that is a real study, not a consolation prize (`03` §3.2, `05` stopping
  conditions).
- Prose in the docs is dense, argued, and opinionated by design. Match that register;
  do not flatten it into bullet-point summary.
- No test suite, linter config, CI, or dependency manifest exists yet. Python 3, stdlib
  only in `src/tacit/` so far.
