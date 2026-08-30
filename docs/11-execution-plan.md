# 11 — Execution plan: end-to-end pass

**Standing instruction from the PI (2026-08-30):** push through to an end result.
Keep the code dead simple and explainable. Avoid unnecessary assumptions at all
costs. **The deliverable is data, not interpretation** — interpretation takes time
regardless, and we will circle back no matter how much is done up front. So: get a
complete pipeline running end to end, produce the panel, and leave the
interpretation for later with the instrument simple enough to change.

## Operating rules for this pass

1. **Data over conclusions.** Every step ends in a file on disk. No step's output
   is a claim.
2. **Simple beats clever.** Explicit lists over inference; substring matches over
   classifiers; a hand-written registry over clustering. Anything a reader cannot
   check by eye is out of scope for this pass.
3. **No unnecessary assumptions.** Where a choice is unavoidable, put it in a data
   file that can be edited, not in code. Where a parameter is unavoidable, sweep
   it (`10`).
4. **The four constraints still bind**: no LLM text assessment (`02` §0), no
   absolute claims (`appendix-deferred` §A), no raw counts (`09`), no arbitrary
   cut points (`10`).
5. **Keep moving.** A blocked step is recorded and skipped, not waited on.

## What is computable from the corpus we have

Layer A + A′ from Crossref, 2006–2025, 40,293 papers. **Titles but no abstracts**,
so every text-based indicator is restricted to title text or unavailable.

| Buildable now | Blocked, and why |
|---|---|
| Technique registry (titles) | `R1`–`R4`, `U1`–`U3`, `B1`, `B4` — need full text |
| Actor graph, lineage proximity (`T1`/`T2` continuous) | `B3` — needs abstracts |
| `B2` reference reach and age | `S1` — needs supplementary/code flags |
| Adoption trajectories, deflated | `F1`–`F4` — need full text |
| `S2` platform mentions (titles only, weak) | `T3` — needs citation *contexts* |
| Institutional measures post-2019 only | `T4`, `H3` pre-2019 — no affiliations |

That is the transfer channel and the epistemic base — enough for a Phase 1 panel.

## Steps

Each produces a file and is independently re-runnable.

| # | Step | Output |
|---|---|---|
| 1 | Technique registry: explicit name+alias list, matched against titles | `data/techniques.json`, `data/technique_papers.csv` |
| 2 | Reference classification: robotics-internal vs external, by venue string and DOI prefix | `data/reference_class.csv` |
| 3 | `B2` reference reach and age, per technique-year | `data/indicator_b2.csv` |
| 4 | Adoption trajectories, deflated by corpus size | `data/adoption.csv` |
| 5 | Lineage proximity per technique, α swept, vs degree-matched null | `data/indicator_lineage.csv` |
| 6 | Assemble the technique × year panel | `data/panel.csv` |
| 7 | Descriptive report over the panel — distributions, not conclusions | `data/report.md` |

## Known limitations to carry forward, not solve now

- Author identity is name-string only: no ORCID, no pre-2019 affiliations
  (`08` F5). Inflates connectivity; worse for East Asian names (`T-H`).
- Technique detection is title-substring only, so recall is bounded by whether a
  paper names its technique in the title. Precision is checkable by hand; recall
  is not, without abstracts.
- Cohort growth confounds lineage proximity (`10` §2). The panel carries adoption
  counts alongside proximity so the confound can be modelled later rather than
  being baked in now.
- 2006 is effectively IROS-only, and 2011 has anomalous reference coverage
  (`08` F5). Flagged in the panel, not dropped.
