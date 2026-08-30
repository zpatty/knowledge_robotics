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

**Status: planning, plus harvest tooling that has never been run against a live API.**
No data has been collected.

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
`xploreapi.py` — an earlier attack on the *same* Ω/λ question by topic modeling, and a
working IEEE Xplore API wrapper. Its commit log records that it "can now scrape IEEE
Xplore for all ICRA and IROS papers." None of the planning documents cite it. Before
building corpus tooling, check what is already there; before treating the IEEE API as
unexercised, note that it has been called from this repository before.

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

## API budget discipline — the code's central concern

The IEEE Xplore key allows **200 calls/day**. One wasted call is 0.5% of a day; one
buggy loop is the whole day. The roles are therefore inverted from the obvious design:

- **OpenAlex is the bulk workhorse** (~650 calls for the full corpus at `per-page=100`).
- **IEEE is authority and gap-filler only** — venue-year completeness counts, abstract
  gap-filling, controlled index terms, supplementary flags.

Three rules are enforced in code, not by discipline:

- `src/tacit/cache.py` — content-addressed on the request URL with credentials
  scrubbed, so a rotated key does not invalidate the cache. Cache-first is the budget
  policy.
- `src/tacit/budget.py` — **reserve-then-call**; the reservation is spent before the
  request goes out, because the API charges for a call whose response fails to parse.
  Persists to `data/budget_ieee.json`; refuses rather than exceeds.
- Every client takes `dry_run=True`. **Always run `--dry-run` first** and read the
  printed URLs.

`IEEEXplore(reserve=N)` holds back N calls so an automated harvest cannot eat the
allowance an interactive probe needs.

### Order of operations

```bash
cp .env.example .env                              # IEEE_API_KEY, OPENALEX_API_KEY, OPENALEX_MAILTO
python3 scripts/probe_ieee.py --dry-run
python3 scripts/probe_ieee.py                     # ~8 calls; determines the whole harvest design
python3 scripts/harvest_openalex.py --stage sources   # inspect the source records BY HAND
python3 scripts/harvest_openalex.py --stage works --sources S... S...
python3 scripts/coverage_report.py                # the Phase 0 gate
```

**Source selection is deliberately manual.** OpenAlex source records for long-running
conference series are split, renamed, and incomplete. Picking one programmatically by
best name match silently truncates the corpus and the failure is invisible. Print the
candidates, look at them, record the chosen IDs in version control.

## Environment

Outbound HTTPS in a default cloud session reaches package registries and GitHub only,
so OpenAlex and IEEE are refused at CONNECT (`HTTP 000`, not a 401 — the request never
leaves the VM). `docs/07-harvest-operations.md` has the fix: set the environment's
network access to **Custom**, allowlist `api.openalex.org`, `ieeexploreapi.ieee.org`,
`export.arxiv.org`, `arxiv.org`, `api.crossref.org`, `api.semanticscholar.org`,
`api.unpaywall.org`, `api.ror.org`, **keep the package-manager defaults checked**, put
the keys in the environment-variables box, and start a new session.

Do **not** use the environment's *API credentials* feature for these keys — it injects
HTTP headers, but both APIs authenticate by query string, so the credential attaches,
is ignored, and produces 401s that look like a bad key.

A cloud VM is the wrong home for the harvest regardless: the container is reclaimed
after the session, and a 65k-work corpus should not be committed. Prefer a persistent
machine for harvesting and cloud sessions for analysis and code.

## What is unverified

Nothing in `src/` or `scripts/` has been run against a live API. Treat every corpus
count in the docs as an estimate. Specifically open:

- Whether the IEEE key is active at all — its granted status reads *waiting*.
- The real `max_records` ceiling (`ieee.py` assumes 200) and whether deep
  `start_record` paging is permitted.
- Which fields IEEE returns per era — abstracts, affiliations, index terms, and
  especially supplementary-material flags, which `S1` leans on heavily.
- OpenAlex abstract coverage before ~2000. **This is the Phase 0 gate**: the
  abstract-coverage-by-year curve is the single most important number to establish.
- Exact IEEE `publication_title` strings across forty years of renamed proceedings.
  `VENUES` in `ieee.py` holds one variant per venue; expect to need several.

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
