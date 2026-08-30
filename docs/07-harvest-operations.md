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

## What is unverified

None of this has been run against the live APIs — the development environment's network
policy denies outbound HTTPS to everything but package registries ([`03`](03-corpus.md)
§3.7). Specifically unverified, in rough order of how much they would change the plan:

- **Whether the IEEE key is active.** Its granted status reads *waiting*.
- **The real `max_records` ceiling** (assumed 200 in `ieee.py`) and whether deep
  `start_record` paging is permitted. An over-large `max_records` that is silently
  truncated would leave gaps that look like missing papers.
- **Which fields the IEEE API returns per era** — abstracts, affiliations, index terms,
  and especially supplementary-material flags, which S1 would lean on heavily.
- **OpenAlex abstract coverage for pre-2000 robotics**, which is the Phase 0 gate.
- **Exact IEEE `publication_title` strings** across forty years of renamed proceedings.
  `VENUES` in `ieee.py` holds one variant per venue; expect to need several.

The probe script exists to answer the first three in one sitting, for eight calls.
