# 08 — Phase 0 findings (live API probe, 2026-08-30)

First contact with the live APIs. Three of the plan's load-bearing premises did not
survive it. This document records what was measured, what it invalidates, and what the
options now are; [`03`](03-corpus.md) and [`06`](06-roadmap.md) have been amended to
match rather than left to contradict it.

Everything below is measured, not inferred. Raw responses are in `data/`.

---

## F1 — The IEEE account is inactive, not merely unapproved

```
HTTP/2 403
x-mashery-error-code: ERR_403_DEVELOPER_INACTIVE
x-error-detail-header: Account Inactive
server: Mashery Proxy
```

The same 403 comes back for a deliberately bogus key, which locates the fault in the
**developer account**, not the credential. No key will work until the account is
activated at IEEE's developer portal. This is the "status reads *waiting*" note in
[`06`](06-roadmap.md) resolved in the unfavourable direction.

**Blocked by this:** the field-availability probe (Phase 0 item 3), the venue-year
completeness audit (item 5), abstract gap-filling, controlled index terms, and the
supplementary/multimedia flags that `S1` was to lean on. It is a human action and
nothing in the codebase routes around it.

`AccountInactive` is now a distinct exception type, so the next person to hit this reads
the cause instead of a bare `403 Forbidden`.

## F2 — OpenAlex is metered now, and the free tier is small

The plan's source architecture rests on OpenAlex being a generous free bulk source
([`03`](03-corpus.md) §3.3: "a comfortable afternoon, not a multi-day operation"). It is
no longer free at that volume. Measured from `x-ratelimit-*` response headers:

| | Value |
|---|---|
| Free daily allowance | **1,000 credits = $0.10** |
| `per-page=1` | 1 credit ($0.0001) |
| `per-page=100` | 5 credits |
| `per-page=200` | 10 credits ($0.0010) |

So credits ≈ `ceil(per_page / 20)`. A ~65k-work harvest at `per-page=100` is ~650 calls
but **~3,250 credits — about 3.3 days of the free allowance**, or roughly **$0.33**
prepaid.

Two consequences, one trivial and one not:

- **The money is irrelevant.** A third of a dollar buys the entire corpus. Top up and
  the constraint disappears. This is the recommended fix.
- **The code was measuring the wrong thing.** `budget.py` counted *calls*, and the
  OpenAlex client was constructed with a 100,000-call daily limit — a guard that would
  never once have fired. Budgeting by call count under-states a paged harvest's true
  spend by 5×. The client is now credit-aware (`credits_for(per_page)`,
  `FREE_DAILY_CREDITS = 1000`), and the local reservation is reconciled against the
  provider's own `x-ratelimit-remaining`.

**The supplied `OPENALEX_API_KEY` is rejected** (`401 "API key not found"`). The keyless
polite pool — `mailto` only — works and is subject to the same daily allowance. The key
is now optional, and a rejected one falls back to keyless once, loudly, rather than
killing the run.

## F3 — OpenAlex does not have the corpus *(the serious one)*

This is the finding that reshapes the study, and it was invisible until the source
records were inspected by hand — exactly the failure [`07`](07-harvest-operations.md)
predicted, though larger than it anticipated.

Enumerating every OpenAlex source record matching each venue name:

**IROS — 4 source records, 3,839 works, spanning three years.**

| Source | Works | Name |
|---|---|---|
| `S4363608614` | 1,577 | 2011 IEEE/RSJ International Conference on Intelligent Robots and Systems |
| `S4363607704` | 1,221 | 2022 IEEE/RSJ … (IROS) |
| `S4363607734` | 1,042 | 2021 IEEE/RSJ … (IROS) |
| `S4393918188` | 0 | Proceedings of the International Conference on Intelligent Robots and Systems |

A `group_by=publication_year` over all four returns **2011, 2021, 2022 and nothing
else**. IROS has run annually since 1988. OpenAlex venue-linked coverage is therefore
roughly **three years out of thirty-five**.

**ICRA — 45 matching source records, of which almost all are other venues.** The
plausible ICRA records total ~2,378 works: `S4363607759` (2022 ICRA, 944),
`S4210217939` (Proceedings — IEEE ICRA, 750, typed *journal*), `S4306419799` (generic
"International Conference on Robotics and Automation", 684). The rest of the 45 are
RA-L, T-RO's predecessor, and a long tail of unrelated automation conferences that the
name search happens to match.

Against an expected ~55–75k papers, **OpenAlex venue linkage delivers under 10%**, and
what it does deliver is clustered in three recent years. The conference series is also
split into **per-year source records**, so there is no single stable source ID to filter
on even for the years that exist.

**Not yet established:** whether the missing papers are absent from OpenAlex entirely,
or present but not linked to a venue source record (`primary_location.source` null or
pointing elsewhere). This distinction decides whether OpenAlex is salvageable, and it is
the single most important open question. It could not be tested — the daily credit
allowance was exhausted mid-investigation.

## F4 — Crossref has what OpenAlex is missing

Probed as an alternative, since it is free, unmetered, and already on the environment's
allowlist. A single spot check for 1995 returns real IROS proceedings articles:

```
container: Proceedings 1995 IEEE/RSJ International Conference on Intelligent Robots …
title:     Nonlinear control of robot manipulators using adaptive fuzzy sliding mode …
year:      1995      reference-count: 12      abstract: absent
```

So Crossref carries **Layer A (venue, title, authors, year) and Layer A′ (reference
edges) for exactly the era OpenAlex cannot reach**. That covers the whole transfer
channel (T1–T4), the substitution channel, and B2 — which is to say the entire Phase 1
study, per [`03`](03-corpus.md) §3.2.

Two caveats:

- **No abstracts.** The sampled records carry none, and IEEE's Crossref deposits are
  inconsistent about them. Abstract coverage is the Phase 0 gate, and Crossref does not
  supply it — that still needs IEEE, or OpenAlex where it has records.
- **`query.container-title` is fuzzy relevance ranking, not a filter.** It returns
  ~7M results for either venue and its `total-results` is meaningless. Enumerating a
  conference properly needs exact container matching (by proceedings ISBN, or a
  member+date query filtered client-side). **Nobody should quote a Crossref count
  obtained the naive way.**

**DBLP is not reachable** — `dblp.org` is not on the environment's allowlist and the
CONNECT is refused. It is the natural authority for conference membership and would be
worth adding to the allowlist alongside the domains in
[`07`](07-harvest-operations.md).

---

## What this does to the plan

The corpus premise in [`03`](03-corpus.md) — OpenAlex bulk, IEEE authority — is dead in
both halves at once. IEEE is inactive; OpenAlex has under 10% of the venue linkage. The
plan's own **stopping conditions** ([`05`](05-validation-and-threats.md)) do not fire:
this is a *source* problem, not a construct problem, and the fallback it names ("pivot
to a Layer A/A′ study on the transfer and substitution channels plus B2") is precisely
what Crossref can still support.

Nothing in the *design* is invalidated. The channels, indicators, hypotheses and
validation plan are untouched. What changed is where Layers A and A′ come from.

### Open decisions

1. **Activate the IEEE developer account.** Everything IEEE-specific waits on it, and it
   is the only route to corpus-wide abstracts and multimedia flags. Human action.
2. **Top up OpenAlex, or don't.** ~$0.33 buys the full corpus pull and removes the
   metering constraint. Cheap enough that the only reason not to is if F3 means OpenAlex
   is not worth harvesting at all — settle F3 first.
3. **Settle F3.** One day's free credits is enough: check whether ICRA/IROS works exist
   in OpenAlex unlinked to a venue source. This decides whether OpenAlex stays in the
   architecture. **Do this before spending anything.**
4. **Promote Crossref from "secondary" to a Layer A/A′ primary** if F3 comes back badly,
   and build exact-container enumeration rather than the fuzzy search used in the probe.
5. **Allowlist `dblp.org`** in the environment; it is the cleanest venue-membership
   authority available and costs nothing to add.

## Defects found and fixed during the probe

None of these were visible before the code met a live API. All are committed with
regression tests (`tests/test_guards.py`, 22 tests).

- **Cache keys depended on parameter order.** Params are passed as keyword arguments, so
  the same logical query spelled in two orders produced two cache entries and was paid
  for twice. Keys are now canonical (`ApiClient.canonical_url` sorts).
- **A 22-hour `Retry-After` would have been slept through.** OpenAlex answers an
  exhausted quota with `429 Retry-After: 79803`. The retry loop would have blocked the
  harvest overnight with no output. Long waits now raise `QuotaExhausted`.
- **The API key leaked into exception messages.** `requests.raise_for_status()` embeds
  the full URL — credential included — in the error, and both APIs authenticate by query
  string. Every error path now scrubs (`cache.scrub_url`). This mattered enough to fix
  properly given the repository has leaked a key once already ([`07`](07-harvest-operations.md)).
- **`budget.py` and `cache.py` had never been executed.** They are the only thing
  standing between a bug and the daily allowance. Now covered: refusal at limit, atomic
  refusal of oversized batches, persistence across restarts, UTC-day rollover, history
  pruning that preserves today, and credential scrubbing in cache keys.
