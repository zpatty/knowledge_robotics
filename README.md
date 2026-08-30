# The Unwritten Half

**Tracing tacit knowledge through the robotics literature (IEEE ICRA + IROS, 1984–present)**

A study, after Joel Mokyr's *The Gifts of Athena*, of the relationship between what
roboticists know how to do (λ-knowledge, prescriptive, technique) and what they can
write down (Ω-knowledge, propositional, explicable) — and of the residue between the
two, which is tacit knowledge.

The central methodological problem is stated in the title of the field's own paradox:
tacit knowledge is by definition the knowledge that does not survive transmission by
text, so it cannot be read off a corpus of texts. This project's wager is that it can
nevertheless be *triangulated* from the corpus, because uncodified knowledge leaves
systematic negative-space traces in the written record — in what papers hedge about,
in who is able to reuse a technique, in what fails to reproduce, in which non-textual
carriers (video, code, benchmarks, platforms, people) get pressed into service, and in
the gap between a method section and an executable specification.

## Read in this order

| Doc | What it settles |
|---|---|
| [`docs/01-framework.md`](docs/01-framework.md) | Definitions. Mokyr's Ω/λ, Collins' tacit taxonomy, and the specific construct we measure: **codification deficit**. |
| [`docs/02-detection-methods.md`](docs/02-detection-methods.md) | **The core deliverable.** Five observation channels and ~20 concrete indicators for detecting tacit knowledge in text, each with an operationalization. |
| [`docs/03-corpus.md`](docs/03-corpus.md) | Building the ICRA/IROS corpus; full-text access strategy; the openness-selection problem. |
| [`docs/04-study-design.md`](docs/04-study-design.md) | Unit of analysis, data model, hypotheses, identification strategies, the headline analyses. |
| [`docs/05-validation-and-threats.md`](docs/05-validation-and-threats.md) | How we know the measures measure anything; what could make the whole thing wrong. |
| [`docs/06-roadmap.md`](docs/06-roadmap.md) | Phases, decision points, a 6-week minimum viable study, deliverables. |
| [`docs/08-phase0-findings.md`](docs/08-phase0-findings.md) | **Read before harvesting.** What the live APIs actually returned, and which premises above it invalidates. |
| [`docs/09-scale-invariance.md`](docs/09-scale-invariance.md) | Why no indicator may rest on a raw count, the measured deflators, and an audit of the battery. |
| [`docs/10-continuous-measurement.md`](docs/10-continuous-measurement.md) | Why no measure may gate a continuous quantity at a cut point, and the continuous lineage measure that replaces T1/T2's distance threshold. |
| [`docs/11-execution-plan.md`](docs/11-execution-plan.md) | The end-to-end pass: what is computable from the corpus we have, and in what order. |
| [`docs/12-data-dictionary.md`](docs/12-data-dictionary.md) | **Every column in every output file**, and what each must not be used for. |

## Status

Planning, with harvest tooling written and first contact made with the live APIs.
No corpus collected yet.

**The corpus premise did not survive that first contact.** The IEEE developer account is
inactive, and OpenAlex — cast as the bulk workhorse — turns out to be metered and to hold
under 10% of the expected ICRA/IROS venue linkage. The study design is unaffected; the
question is where Layers A/A′ come from. See
[`docs/08-phase0-findings.md`](docs/08-phase0-findings.md), which supersedes the source
architecture in §3.3 of the corpus document.

Open decisions are at the end of [`docs/06-roadmap.md`](docs/06-roadmap.md) and in §
*Open decisions* of the findings document.
