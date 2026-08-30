# 12 — Data dictionary

Every column in every file the pipeline produces, what it means, and what it must
not be used for. Written so the outputs are usable without reading the code.

Files under `data/` are gitignored and regenerable; curated inputs under
`registry/` are version-controlled. Regenerate everything with:

```bash
python3 scripts/harvest_crossref.py --years 2006-2025   # ~2 min, resumable, cached
python3 scripts/build_panel.py
python3 scripts/build_lineage.py --alphas 0.15          # hours, resumable
python3 scripts/corpus_baselines.py
python3 scripts/audit_registry.py
python3 scripts/report.py
```

---

## `registry/techniques.json` — curated input

The unit of analysis (`04` §1). Hand-written, edit freely.

| Field | Meaning |
|---|---|
| `id` | Stable key used in every output. Do not renumber. |
| `name` | Human label. |
| `axis` | `algorithm` / `hardware-coupled` / `hardware`. A **label carried to the panel, never an input to a measure** — it exists so D1's predicted gradient can be tested, not to compute anything. |
| `aliases` | Lowercased substrings matched against the padded title. Pad short ones (`" mpc "`) so they match words. |
| `exclude` | Substrings that veto a match. Use for known false positives. |

## `registry/reference_classes.json` — curated input

B2's classifier (`02` §6). DOI structure first, venue text as fallback.

| Field | Meaning |
|---|---|
| `ieee_stems` | Alphabetic part of an IEEE DOI suffix → field. `10.1109/ICRA40945.2020` reduces to `ICRA`. Numeric stems are IEEE publication numbers (`70` = T-RA, `34` = TPAMI, `9` = TAC). |
| `registrant_prefixes` | DOI registrant → field (`10.15607` = RSS). |
| `doi_prefixes` | Longer DOI prefixes (`10.1177/0278364` = IJRR). |
| `venue_markers` | Lowercased substrings of the reference's venue string. Fallback only. |

---

## `data/crossref_works.jsonl` — the corpus

One JSON object per paper. 39,839 papers, 2006–2025, ICRA 18,837 / IROS 21,002.

| Field | Notes |
|---|---|
| `venue` | `ICRA` or `IROS`. Contamination is zero after the tail rule in `crossref.py`. |
| `year` | From the record's own date fields, preferring publication over deposit (`08` F4). |
| `doi`, `title`, `container`, `type`, `page`, `publisher`, `isbn`, `license` | As deposited. |
| `authors[]` | `name`, `family`, `orcid`, `sequence`. **`orcid` is present on 30 papers in 40k** — treat as absent (`08` F5). |
| `affiliations[]` | Flat list of strings. **Absent before 2019, ~98% from 2022.** Not a clean changepoint (`08` F4). |
| `references[]` | `doi`, `year`, `title`, `venue`, `unstructured`. 69% carry a DOI, 29% a venue string. |
| `n_references`, `n_refs_with_doi`, `n_authors`, `n_affiliations` | Convenience counts. **Counts, not indicators** — never use one as a measure without a denominator (`09`). |

## `data/panel.csv` — technique × year

748 cells. The main analysis table.

| Column | Meaning and cautions |
|---|---|
| `technique`, `technique_name`, `axis` | From the registry. |
| `year` | Publication year. |
| `papers` | Count of papers matching this technique this year. **Lower bound**: title matching misses papers that do not name the technique in the title, and the miss rate is unmeasurable without abstracts. |
| `share_of_year` | `papers` ÷ all corpus papers that year. **Use this, not `papers`**, for anything comparative (`09`). |
| `n_authors` | Distinct authors on those papers. |
| `share_of_year_authors` | `n_authors` ÷ all distinct corpus authors that year. The deflated form. |
| `entrant_share` | Fraction of this technique-year's authors appearing anywhere in the corpus for the first time. A crude proxy for cohort turnover; the confound in `10` §2 lives here. |
| `b2_reach` | External ÷ classified references (`02` §6, B2). Empty when nothing classified. |
| `b2_classified_share` | Fraction of references the classifier could place. **Read this before `b2_reach`** — a reach figure resting on 20% of a reference list is a different object from one resting on 70%. |
| `ref_age_mean`, `ref_age_median` | Citing-minus-cited year gaps. The more robust half of B2, since it needs no classification. |
| `refs_per_paper` | Context for the deflation problem: this roughly doubles across the window corpus-wide. |
| `top_external_field` | Modal external field cited. Descriptive. |

## `data/panel_corpus.csv` — the denominators

`year`, `papers`, `distinct_authors`, `refs_per_paper`. The values every panel
ratio is computed against. Keep it alongside the panel so ratios are auditable.

## `data/indicator_lineage.csv` — continuous T1/T2

One row per technique × year × α. Rolling: each year's *new* adopters against all
prior authors of that technique, with no cohort split (`10` §2).

| Column | Meaning |
|---|---|
| `technique`, `year`, `alpha` | α is the PPR restart probability — a decay parameter. **Sweep it; never report one value** (`10`). |
| `n_prior_authors`, `n_new_authors` | Cohort sizes. **Read these before `ratio`.** |
| `observed` | Adopters' share of personalised-PageRank mass from the prior-author seeds. |
| `expected` | Same quantity from degree-matched random seeds, averaged over `n_null` draws. |
| `ratio` | `observed ÷ expected`. **1.0 = no closer than similarly-connected strangers.** The scale-free form. |
| `z` | `(observed − expected) ÷ sd(null)`. A ratio of 1.4 on a tight null and a wide one are different findings. |
| `n_null` | Null draws. `0` marks a cell recorded as too small to compute — kept in the file rather than dropped, so the skip is visible. |
| `seconds` | Wall time. Diagnostic. |

**The standing caution.** This column is **not** a tacitness ranking. Note the
cohort-growth confound predicted in `10` §2 is measured at Spearman +0.081 over
the 688 computed cells and is therefore negligible *in this rolling design* — the
warning applied to the 2015-split probe, not to the panel. What remains open is
whether growth matters within a single technique's trajectory, and, more
importantly, that nothing yet connects this ratio to the construct: it measures
network proximity, and calling that tacit transmission is an inference the data
does not yet support.

## `data/corpus_baselines.json` — deflators

Per year: `papers`, `refs_per_paper_mean` / `_median`, `authors_per_paper`,
`distinct_authors`, `coauthor_edges`, `mean_degree`, `mean_degree_fractional`,
`with_references`, `with_affiliation`, `authors_median` / `_p90` / `_p99` / `_max`.

Use `mean_degree_fractional`, not `mean_degree`: the raw form inflates ×1.73
across the window and a single 279-author paper drives 2024 to 12.67 (`09` §2).

## `data/registry_audit.md` — for reading, not parsing

Per technique: matched count, per-alias match counts, a `sole-carrier` column
(papers no sibling alias also matches — where one loose string carries a technique
alone), and a random sample of titles. Flags aliases matching >2% of the corpus.
**Nothing here filters anything.**

## `data/report.md` — descriptive output

Regenerated by `scripts/report.py`. Contains no inferences by construction.

---

## What is not in any of these files

Because the source cannot supply it (`08`, `11`):

- **Abstracts**, so `R1`–`R4`, `U1`–`U3`, `B1`, `B3`, `B4`, `F1`–`F4` are all
  unbuilt. Needs IEEE.
- **1984–2005**, so no forty-year series and H5a/H5b stay underpowered. Needs
  IEEE or DBLP for per-venue-year DOI lists (`08` F4).
- **Code/video/supplementary flags**, so `S1` is unbuilt and `S4` uncurated.
- **Citation contexts**, so `T3` is unbuilt.
- **Affiliation histories before 2019**, so `T4` and `H3` are unbuildable over any
  useful window.
