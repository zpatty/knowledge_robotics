# 08 — Phase 0 findings (live API probe, 2026-08-30)

First contact with the live APIs. Every source premise in the plan failed, in a
different way each time. This document records what was measured, what it invalidates, and what the
options now are; [`03`](03-corpus.md) and [`06`](06-roadmap.md) have been amended to
match rather than left to contradict it.

Everything below is measured, not inferred. Raw responses are in `data/`.

---

## F1 — IEEE blocks this caller before authentication (not an account problem)

```
HTTP/2 403
x-mashery-error-code: ERR_403_DEVELOPER_INACTIVE
x-error-detail-header: Account Inactive
server: Mashery Proxy
```

**Revised 2026-08-30, second pass.** The first reading of this was "the developer
account is inactive". The evidence does not support it, and the corrected
diagnosis matters because it changes who has to do something.

### The gateway never evaluates the key

Every key shape returns a byte-identical response:

| Request | Result |
|---|---|
| The real 24-char key | `ERR_403_DEVELOPER_INACTIVE` |
| A wrong 24-char key | `ERR_403_DEVELOPER_INACTIVE` |
| A 3-character key | `ERR_403_DEVELOPER_INACTIVE` |
| An 80-character key | `ERR_403_DEVELOPER_INACTIVE` |
| Special characters | `ERR_403_DEVELOPER_INACTIVE` |
| `apikey=` (empty) | `ERR_403_DEVELOPER_INACTIVE` |
| **No `apikey` parameter at all** | `ERR_403_DEVELOPER_INACTIVE` |

A Mashery gateway that were checking credentials would distinguish *missing* from
*malformed* from *unknown* from *inactive* — those are separate error codes in the
product. Returning one code for all seven cases means the request is refused
**before authentication runs**.

Meanwhile routing works: `/api/v1/` and the gateway root return
`ERR_596_SERVICE_NOT_FOUND`, and both `search/articles` and
`search/document/{n}/fulltext` resolve to real services. So the gateway knows the
path and rejects the caller, not the credential.

### The caller is a datacenter IP, twice over

The same 403 arrives over two independent network paths with different source
addresses:

| Path | Egress IP | Owner |
|---|---|---|
| HTTPS via the session proxy | `160.79.106.128` | Anthropic |
| Plain HTTP, proxy bypassed | `34.122.140.224` | Google Cloud |

Both are cloud ranges. Publishers routinely block datacenter address space ahead
of authentication to stop bulk scraping, and that hypothesis fits every
observation here: key-independent, path-sensitive, reproducible from two
unrelated cloud IPs.

**So the most likely reading is that the key is fine and the origin is the
problem.** "Account Inactive" is Mashery's generic pre-auth refusal string, not a
statement about this account.

### The test that settles it

Run the exact request from the network the key was registered on — a campus or
home connection, not a VM or VPN:

```bash
curl -sS -D - -o /dev/null \
  "https://ieeexploreapi.ieee.org/api/v1/search/articles?querytext=robot&max_records=1&apikey=YOUR_KEY"
```

- **200 with JSON** — the key works and the block is our cloud origin. Harvest
  from that machine; nothing needs to be asked of IEEE. This is what
  [`07`](07-harvest-operations.md) already recommends for other reasons.
- **The same `ERR_403_DEVELOPER_INACTIVE`** — then it is account-side after all,
  and IEEE support is the route. Quote the error code and the fact that it
  reproduces with no key present.

If it works there, the pre-2007 corpus and the abstract layer are both
recoverable without any negotiation — which would change the project's
sequencing considerably, since they are the two things currently blocking
H5a/H5b and the entire linguistic channel.

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

## F4 — Crossref reaches back only to ~2007, and it is a hard edge

Crossref was probed as the alternative: free, unmetered, already allowlisted. It
carries what the plan needs — container title, article title, DOI, year, authors,
**the full `reference` array** (not merely a count), and affiliations — but only for
part of the period.

Measured ICRA+IROS papers recovered per year (`scripts/survey_crossref.py`):

| Year | ICRA | IROS | Total | with refs | with affiliations |
|---|---|---|---|---|---|
| 1995 | — | — | 74 | — | — |
| 2002 | — | — | 38 | — | — |
| 2006 | 29 | 1,019 | 1,048 | 994 | 30 |
| 2007 | 808 | 699 | 1,507 | 1,492 | 21 |
| 2008 | 670 | 690 | 1,360 | 1,333 | 23 |
| 2010 | 878 | 995 | 1,873 | 1,862 | 23 |
| 2015 | 940 | 971 | 1,911 | 1,896 | 46 |
| 2018 | 868 | 1,100 | 1,968 | 1,931 | 45 |
| 2022 | 941 | 1,220 | 2,161 | 2,139 | **2,125** |
| 2024 | 1,889 | 1,587 | 3,476 | 3,452 | **3,421** |

Two edges, and they sit in different places:

- **Papers: usable from ~2007, empty before.** 1995 and 2002 return dozens where
  they should return a thousand or more. ICRA 2006 (29 papers) is a hole inside
  the good era and needs its own look.
- **Reference edges: ~99% throughout the covered years.** Layer A′ is in excellent
  shape wherever Layer A exists, which is the single most encouraging result of
  the probe — the transfer channel (T1–T4) and B2 run on exactly this.
- **Affiliations: a ramp from 2019, not a clean cliff.** Sparse sampling
  suggested a step at 2022; the full harvest shows 0–3% through 2018, then 44%
  (2019), 12% (2020), 22% (2021), and 98–99% from 2022. The 2019 spike and the
  2020 fallback mean this is inconsistent deposit practice rather than a policy
  date, so any affiliation-based measure must be conditioned on year and
  cannot assume a single changepoint.

That last point bites specific indicators. **T4** (personnel-flow-mediated
diffusion) and **H3** (institutional and geographic decentralisation) are built on
affiliation histories and cannot be computed from Crossref before 2022 — which is
far too short a window for either. T1 and T2 depend on co-authorship rather than
institution and survive, since author names are present throughout.

### The records exist but are not enumerable

Worth stating precisely, because it changes what a fix would look like.
`10.1109/robot.2002.1013340` — a real ICRA 2002 paper — resolves perfectly by
direct DOI lookup, carries member 263, type `proceedings-article`, and
`published: 2002`. Yet enumerating exactly that filter for 2002 returns 9,673
works containing only 21 `robot.*` DOIs. The pre-2007 corpus is *in* Crossref and
merely unreachable through its search index. IEEE re-deposited this era in 2022
and it was indexed in 2025; that timing is the likely cause.

**So a DOI list would unlock the old years.** Crossref can retrieve what it cannot
enumerate. Anything that supplies per-venue-year DOI lists — IEEE Xplore, DBLP —
converts the pre-2007 corpus from unreachable to a straightforward fetch.

### Three Crossref traps, all silent

Recorded because each cost real time and none produces an error:

1. **`select` drops date fields.** `select=published` returns records with no
   `published` key rather than an error, so every year comes back undated. An
   *invalid* select name is rejected loudly; a valid-but-unsupported one is not.
2. **`query.container-title` is fuzzy ranking, not a filter.** It returns hundreds
   of thousands of results for either venue; its `total-results` is meaningless.
3. **Cursor paging discards relevance order.** The expensive one. A `query.*`
   search is relevance-ordered only on an un-cursored request; `cursor=*`
   re-orders by an internal key, so paging a fuzzy query walks arbitrary records.
   On ICRA 2015 the first un-cursored page is 200/200 genuine ICRA while six
   cursor-paged pages over the same query return **zero**. An earlier draft of
   this document quoted 4–8% match rates measured the wrong way; those numbers
   were an artifact of this and have been removed. Use `offset` with a fuzzy
   query, `cursor` only with a pure filter.

**DBLP is not reachable** — `dblp.org` is not on the environment's allowlist and
the CONNECT is refused. Given that a DOI list is what unlocks the old corpus, and
DBLP is the cleanest free source of one, adding it is now the highest-value change
to the environment configuration.

### Original spot check, kept for the record


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

## F5 — The harvested corpus, validated

`scripts/harvest_crossref.py` over 2006–2025 produced **40,293 papers**, every one
with a distinct DOI. This is the working corpus.

| | |
|---|---|
| Papers | 40,293 (ICRA 19,291 · IROS 21,002) |
| Reference edges | **981,672**, of which 723,819 (74%) DOI-resolved |
| Papers with ≥1 reference | 39,206 (97%) |
| Distinct author names | 68,576 |
| Papers with any affiliation | 13,248 (33%, almost all post-2019) |
| Papers with any ORCID | **30** |

Four things in that table change how the study should be built.

**Layer A′ is strong — stronger than the early spot check suggested.** 74% of
reference entries carry a DOI, not the 47% measured on a single 2002 paper. Nearly
a million resolvable citation edges is a real graph, and T1/T2/T3 and B2 run on it
directly. The remaining 26% are unstructured strings that need matching before they
can be used; that bounds recall rather than blocking anything.

**ORCID is effectively absent.** Thirty papers out of forty thousand. The plan's
author-disambiguation strategy ([`03`](03-corpus.md) §3.5) puts ORCID first and an
ensemble second; in practice there is no ORCID layer at all, so disambiguation rests
entirely on name + affiliation + co-author + topic signals — and affiliation is
itself missing before 2019. This makes **T-H** (disambiguation error, worse for
non-Western names) materially more severe than
[`05`](05-validation-and-threats.md) assumed, and the hand-checked sample it calls
for is now mandatory rather than good practice. The ORCID-only robustness check
proposed there is not available.

**References per paper more than double across the window** — 14.6 (2006) to 33.9
(2025), rising steadily. B2 is specified as a *fraction* of citations that reach
outside robotics, so it is partly insulated, but a doubling of the denominator over
the same period as the outcome is a structural trend that has to be modelled
explicitly, not assumed away. It also interacts with **T-A** (writing-convention
drift): reference-list growth is exactly the kind of venue-norm change that
masquerades as a knowledge trend.

**Two year-level anomalies** need a look before the panel is used:
- **2011: 2,521 papers but only 71% carry references**, against 97–100% in every
  neighbouring year. Both numbers are out of line.
- **2006: ICRA returns 29 papers** against IROS's 1,019. ICRA 2006 is essentially
  missing, so 2006 should be treated as IROS-only or dropped.


### Standing options (PI, 2026-08-30)

Two levers are available and should be assumed available in any future decision
here, rather than re-litigated:

1. **Paying for OpenAlex is acceptable**, within reason and not open-ended. Note
   the exposure is bounded by construction — OpenAlex sells *prepaid credits*, not
   a subscription: when the balance is spent requests 403 rather than billing on.
   A full-corpus pull is ~$0.33. Cost is therefore never a reason to avoid an
   OpenAlex route; only coverage (F3) is.
2. **IEEE cooperation is probably obtainable** through the PI's own networking,
   at the cost of some lead time. This changes the calculus on F1 and on the TDM
   licence: routes that depend on IEEE goodwill are worth *planning for* rather
   than designing around, provided nothing on the critical path blocks while the
   ask is in flight.

The practical consequence for sequencing: build what is reachable without either
lever (the Crossref window), and put the IEEE ask in flight in parallel, since it
is the only thing that recovers 1984–2006 and the abstract layer.

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
