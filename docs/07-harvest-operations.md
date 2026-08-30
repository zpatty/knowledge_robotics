# 07 — Harvest operations

Operational notes for the code in `src/tacit/` and `scripts/`. The design is shaped
almost entirely by one number: **200 IEEE calls per day.**

## Budget arithmetic

| Source | Limit | Full-corpus cost | Role |
|---|---|---|---|
| OpenAlex | generous daily budget (free key = 10× keyless), 100 req/s ceiling | ~650 calls at `per-page=100` for ~65k works | **bulk workhorse** |
| IEEE Xplore | **200 calls/day**, 10/s | ≥ 2 days at best case, 3–5 realistically | **authority + gap-filler** |

One wasted IEEE call is 0.5% of a day. One buggy loop is the whole day. Three
consequences, all enforced in code rather than by discipline:

1. **Cache-first is the budget policy, not an optimization.** `cache.py` is
   content-addressed on the request URL with credentials scrubbed, so a rotated key does
   not invalidate the cache and a re-run of a completed harvest costs nothing.
2. **Reserve-then-call.** `budget.py` spends the reservation *before* the request goes
   out, because the API charges for a call whose response we then fail to parse. A 429
   retry does not re-spend — the reservation already covers it. State is persisted to
   `data/budget_ieee.json` so it survives process restarts, and it refuses rather than
   exceeds.
3. **Dry-run before spend.** Every client takes `dry_run=True` and prints the exact URLs
   it would request. Review the plan, then run it.

`IEEEXplore(reserve=N)` holds back N calls so an automated harvest cannot eat the
allowance an interactive probe needs.

## Order of operations

```bash
cp .env.example .env          # fill in keys; .env is gitignored

# 1. Probe — ~8 IEEE calls. Determines everything downstream.
python3 scripts/probe_ieee.py --dry-run
python3 scripts/probe_ieee.py

# 2. Find venue sources — inspect by hand, do not automate the choice.
python3 scripts/harvest_openalex.py --stage sources

# 3. Bulk harvest from OpenAlex.
python3 scripts/harvest_openalex.py --stage works --sources S... S...

# 4. The Phase 0 gate.
python3 scripts/coverage_report.py
```

## Why the source-selection step is manual

OpenAlex source records for long-running conference series are messy: renamed
proceedings, split records, per-year records, missing years. Selecting one source ID
programmatically by best name match will silently truncate the corpus, and the failure is
invisible — you get a plausible-looking corpus that is missing 1991–1996. Print the
candidates, look at them, record the chosen IDs in version control, and cross-check the
per-year counts against the IEEE completeness audit.

## Running this in a Claude Code cloud environment

The default **Trusted** network level reaches package registries and GitHub only, so
OpenAlex and IEEE are refused at CONNECT (`HTTP 000`, not a 401 — the request never
leaves the VM). To fix, edit the environment from the selector at claude.ai/code (the
cloud icon above the message box; there is no settings page for it), then:

1. **Network access → Custom**, and in **Allowed domains**, one per line:
   ```
   api.openalex.org
   ieeexploreapi.ieee.org
   export.arxiv.org
   arxiv.org
   api.crossref.org
   api.semanticscholar.org
   api.unpaywall.org
   api.ror.org
   ```
2. **Check "Also include default list of common package managers."** Unchecked, the list
   above becomes the *only* reachable set and `pip install` breaks.
3. Put the keys in the environment's **Environment variables** box in `.env` format
   (`IEEE_API_KEY`, `OPENALEX_API_KEY`, `OPENALEX_MAILTO`). The repo's `.env` is
   gitignored, so a fresh cloud session clones without it; `config.py` reads real
   environment variables first, so this is the durable place for them.
4. **Start a new session.** Running sessions do not re-read environment configuration.

Do **not** use the environment's *API credentials* feature for these two keys. It injects
HTTP headers, and both APIs authenticate by query string (`?apikey=` for IEEE,
`?api_key=` for OpenAlex) — the credential would attach and be ignored, producing 401s
that look like a bad key. GitHub needs no allowlist entry; it uses a separate proxy.

**A cloud VM is the wrong home for the harvest anyway.** The container is reclaimed after
the session, and a 65k-work corpus dump should not be committed to the repo. Prefer
running the harvest on a persistent machine and using cloud sessions for analysis and
code.

## What the 2020 code on `master` already settles

The `master` branch carries an earlier pass at this corpus (see the `CLAUDE.md` note on
branch topology): `xploreapi.py`, which is IEEE's own distributed Python sample client,
and a notebook that calls it. Its commit log reports a successful full scrape of both
venues. The code is not worth reviving — it is a MALLET/LDA topic-modeling pipeline on
a different question — but it does retire three of the unknowns below for free, and it
is authoritative in a way a guess is not:

- **`max_records` really is capped at 200.** IEEE's client hardcodes
  `resultSetMaxCap = 200` and *clamps* larger requests rather than rejecting them. This
  is the dangerous failure mode the plan worried about: an over-large request returns
  200 records and no error, so a harvest built on a larger step would page past records
  it never fetched. `MAX_RECORDS` in `ieee.py` is now stated as confirmed.
- **Deep `start_record` paging is permitted.** The 2020 notebook pages
  `start_record = i*200 + 1` across an entire venue's history in one loop.
- **The venue title strings carry no `IEEE` prefix.** The successful run used
  `International Conference on Robotics and Automation` and
  `International Conference On Intelligent Robots and Systems` — *not* the
  `IEEE …` / `IEEE/RSJ …` forms this repo's first draft assumed. `VENUES` in `ieee.py`
  now uses the empirically-working strings, with `VENUE_TITLE_VARIANTS` holding the
  alternates for the probe to try.

Two further lessons, both taken into `ieee.py`:

- **Pin the sort when paging.** IEEE's client defaults to
  `sort_field=article_title&sort_order=asc`. Ours set no sort at all, which leaves
  page-boundary ordering unspecified and can silently skip or duplicate records across
  a multi-call harvest. Now pinned.
- **The 2020 loop is a worked example of the failure this repo's budget guard exists to
  prevent.** It reads `total_records` once, floor-divides by 200, and then fetches the
  remainder with a *hardcoded* count (`161`) — correct for one venue on one day and
  wrong forever after, and every retry of it costs the full daily allowance.

**Note also that the 2020 code paged the entire corpus in a single run**, which either
predates the 200/day limit or exceeded it. Do not read it as evidence that the current
key can do the same.

## What is still unverified

Nothing in `src/` or `scripts/` has been run against the live APIs — the development
environment's network policy denies outbound HTTPS to everything but package registries
([`03`](03-corpus.md) §3.7). Specifically unverified, in rough order of how much they
would change the plan:

- **Whether the current IEEE key is active.** Its granted status reads *waiting*. (The
  2020 key is a separate credential and must not be reused — see the security note
  below.)
- **Which fields the IEEE API returns per era** — abstracts, affiliations, index terms,
  and especially supplementary-material flags, which S1 would lean on heavily.
- **OpenAlex abstract coverage for pre-2000 robotics**, which is the Phase 0 gate.
- **Whether the venue title strings above cover all forty years.** They worked for a
  bulk pull in 2020, but IROS ran as a *Workshop* before 1989 and proceedings titles
  were renamed more than once; the per-venue-year completeness audit is what catches a
  variant that silently returns zero for an era.

The probe script answers the field-availability question in one sitting, for eight
calls. With the ceiling and paging questions now settled from `master`, its ceiling test
is a cheap confirmation rather than a live unknown.

## Security note: the 2020 API key is exposed

`robotics_knowledge.ipynb` on the `master` branch has an IEEE Xplore API key hardcoded
as a string literal, in three cells, in a **public** repository — so it should be
treated as compromised regardless of whether it is still active. **Revoke it at the IEEE
developer portal.** It is unrelated to the current key, which is why nothing in this
repo's tooling depends on it.

Rewriting `master`'s history to purge the value is optional and mostly theatre once the
key is revoked; revocation is the part that matters. The current design keeps
credentials in `.env` (gitignored) or the environment, read via `config.require()`, and
`cache.py` scrubs them from cache keys — so this class of leak should not recur.
