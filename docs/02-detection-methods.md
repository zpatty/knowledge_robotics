# 02 — Detecting tacit knowledge in text

**This is the study's core methodological contribution.** The problem: tacit knowledge is
defined by its absence from text, so no amount of reading finds it directly. The
resolution: uncodified knowledge is not *inert*. It deforms the record around itself. Six
channels carry that deformation into the corpus, and each yields indicators we can
compute.

| # | Channel | The trace it leaves | Detects |
|---|---|---|---|
| 1 | **Repair** | Text visibly straining to codify craft: hedges, tricks, warnings, "in practice". | RTK at the moment of conversion |
| 2 | **Transfer** | Techniques that travel with people rather than with papers. | RTK, STK, MTK |
| 3 | **Failure** | Reproduction gaps, sim-to-real gaps, reported non-replications. | All kinds |
| 4 | **Substitution** | Non-textual carriers pressed into service: video, code, benchmarks, platforms, competitions, acknowledgments. | STK, MTK, CTK |
| 5 | **Underdetermination** | The gap between a method section and an executable specification. | RTK (sharply) |
| 6 | **Epistemic base** | Techniques justified by "it worked" rather than by theory. | Mokyr's B(λ,t); context for all of the above |

Each indicator below is specified as: **what it is → how to compute it → what it costs →
what it confounds with**. Indicator IDs (`R1`, `T2`, …) are referenced throughout the
other documents.

---

## 1. Repair channel — codification caught in the act

When an author writes "in practice we found it necessary to…", they are performing
codification: converting something learned by doing into something readable. These
sentences are the frontier between λ and text. Their density, their content, and their
*location in the document* are all informative.

### R1 — Craft-advice density
**What:** rate of sentences performing craft transmission rather than claim-making.
**How:** sentence-level classifier over full text, restricted to Method / Implementation /
Experiment sections. Seven target classes, each with distinct theoretical meaning:

| Class | Marker examples | Reading |
|---|---|---|
| Tuning advice | "we found α = 0.3 worked best", "requires careful tuning" | RTK being partially transferred |
| Caveat / boundary | "this assumes", "fails when", "only valid if" | RTK *and* evidence of a real epistemic base |
| Trick | "the key trick is", "we use the standard trick of" | RTK, often community-shared (CTK-adjacent) |
| Experiential attribution | "in our experience", "we observed that", "empirically" | Appeal to unwritten authority |
| Failure note | "did not converge", "we were unable to", "this did not work" | Negative knowledge surfacing |
| Manual intervention | "by hand", "manually", "human operator resets" | STK / MTK |
| Necessity-without-reason | "it is essential to", "must be", with no justifying clause | RTK, thin base |

**Build:** start with a seed lexicon (~300 patterns) for recall, hand-label a stratified
sample of 3–5k sentences, fine-tune a small encoder (DeBERTa-v3 or ModernBERT) or few-shot
an LLM with the labeled set as exemplars. Report per-class κ against two human annotators.
**Cost:** low-moderate. One annotation round; inference is cheap.
**Confounds:** *This is the most confounded indicator in the battery.* Writing style,
venue norms, page limits, non-native-English authorship, and post-2023 LLM-assisted
writing all move it. It must never be used alone, must be normalized within
year × subfield × page-count strata, and should be reported alongside a style-only
placebo (see [`05`](05-validation-and-threats.md) §3).

### R2 — Codification-effort locus
**What:** *where* craft advice lives in the document. Migration from body text → footnotes →
appendices → supplementary material → external repositories is the observable trajectory
of knowledge being pushed out of the canonical text.
**How:** tag each R1 hit with its structural position. Track the distribution over time and
by subfield. A field whose craft advice has all migrated to a linked repo has *codified* it;
a field whose craft advice is vanishing without a destination has *dropped* it.
**Cost:** low, given parsed document structure (GROBID gives section hierarchy).
**Confounds:** venue supplementary-material policies changed over the corpus period; must
be modeled as a venue-year fixed effect.

### R3 — Unexplained-parameter density
**What:** numeric constants that appear with no derivation, no units-based reasoning, and
no sensitivity analysis. A gain of 0.03 justified by nothing is pure λ on a bare base.
**How:** extract numeric literals with surrounding context; classify each as
*derived* (follows from a stated model/physical quantity), *measured*, *cited*, *swept*
(a sensitivity analysis or ablation exists), or *bare*. Indicator = bare / total.
Normalize by method-section length.
**Confounds:** deep-learning papers have vastly more hyperparameters by construction —
must be compared within method families, not across.

### R4 — Acknowledgment structure
**What:** acknowledgments to technicians, machinists, lab engineers, and to named
individuals "for help with the hardware / for assistance with the setup / for advice on".
These are receipts for tacit transfer that happened outside the text.
**How:** NER + role classification over acknowledgment sections. Distinguish funding
acknowledgments (uninformative) from personal-assistance acknowledgments (informative).
Rate of personal-assistance acknowledgments per paper, by subfield, over time.
**Cost:** low. **Confounds:** acknowledgment norms vary by lab culture and country.
Underused and, I suspect, unusually clean — worth an early look.

---

## 2. Transfer channel — knowledge that travels with people

This is the Collins TEA-laser test, made quantitative. If a technique diffuses only along
lines of personal contact, its executable content is not in the papers. If it diffuses to
strangers who only ever read it, it is codified. **This is the strongest indicator family
in the battery** — it is behavioral rather than linguistic, it is robust to writing style,
and it is computable from metadata alone (so it works even where full text is unavailable).

### T1 — Independent Adoption Latency (IAL)
**What:** time from a technique's originating publication to its first *successful,
independent* use — by a group with no co-authorship path of length ≤ 2 to any originator,
and no shared author.
**How:** requires the technique registry ([`04`](04-study-design.md) §1) and an adoption
classifier that separates *citing* from *using* (see T3). Compute IAL per technique;
model as a survival/hazard problem with censoring for never-adopted techniques.
**Confounds:** technique salience and field size drive adoption speed independently of
tacitness. Control by matching on citation trajectory, subfield, and originator prestige;
prefer within-technique-over-time and within-subfield comparisons.

### T2 — Lineage ratio
**What:** fraction of the first *k* independent-lab adoptions that are lineage-connected
(co-authorship distance ≤ 2, or a known advisor/postdoc relationship). High = knowledge
moving through bodies.
**How:** build the actor graph — co-authorship from the corpus, plus advisor/student edges
where recoverable (thesis metadata, ProQuest, the Mathematics/CS Genealogy projects, lab
web pages). Personnel movement is inferable from affiliation change across a person's
publication sequence.
**Note:** T1 and T2 are the pair that lets us make the study's cleanest causal statement.
See the diff-in-diff in [`04`](04-study-design.md) §4, H2.

### T3 — Citation/implementation divergence
**What:** techniques cited far more often than they are *used*. A method everyone cites and
nobody reimplements is a method whose paper does not contain it.
**How:** classify each citation context into: background mention / comparison baseline /
*actual use* / extension. Citation-intent classification is a solved-enough task
(SciCite-style models); fine-tune on ~2k robotics-specific labels. Indicator =
use-citations / total citations, with baseline-comparison citations tracked separately
(being used as a baseline usually implies a working reimplementation *or* a released one —
which of the two is itself informative).
**Confounds:** foundational results are cited ritually; restrict to techniques within a
comparable maturity band.

### T4 — Personnel-flow-mediated diffusion
**What:** does a technique appear at institution *j* only after a person from institution
*i* arrives there?
**How:** event-history model. For each (technique, institution) pair, hazard of first
adoption as a function of (a) time since publication, (b) arrival of a person with prior
technique experience, (c) release of a codification artifact. The relative magnitude of
(b) versus (c) is a direct estimate of how much of the technique rides in people versus
in text.
**Cost:** high — requires author disambiguation and affiliation histories. But this is the
single most Mokyr-faithful measurement in the design, and it is worth the cost.
**Confounds:** author name disambiguation errors, which are severe for common names and
worse for non-Western names; use ORCID where present and report disambiguation error
rates from a hand-checked sample.

---

## 3. Failure channel — where the deficit is admitted

### F1 — Reported reproduction failures
**What:** explicit statements that a prior method could not be reproduced, or could only be
reproduced after contacting authors, or performed worse than reported.
**How:** high-precision pattern extraction over citation contexts ("we were unable to
reproduce", "our reimplementation of [X] achieves", "using the authors' code", "after
correspondence with the authors"). Each hit is a *directed edge* from the reproducing paper
to the failed technique — this yields a sparse but very high-quality labeled set for
validating everything else.
**Cost:** low to extract, and disproportionately valuable. Prioritize.

### F2 — Reimplementation performance gap
**What:** when paper B reimplements technique A and reports a number, how far below A's
reported number does it land? The gap is a direct estimate of D in performance units.
**How:** extract (technique, benchmark, metric, value, reporting paper) tuples from result
tables; compare originator-reported to independent-reported values on matched benchmarks.
**Confounds:** competitive incentive to under-tune baselines. Partially separable: compare
gaps for baselines against gaps in dedicated reproduction studies.

### F3 — Sim-to-real gap magnitude
**What:** robotics states its own tacit residual out loud. Where a paper reports both
simulated and physical results, the gap is the portion of the technique that the model —
i.e. the epistemic base — fails to capture.
**How:** paired sim/real result extraction. Track the population distribution of gaps by
subfield and year. **This is a rare case where the field has already instrumented its own
codification deficit; exploit it.**
**Confounds:** reporting bias (large gaps go unreported). Bound with a selection model;
compare against competition results where reporting is compulsory.

### F4 — Trial-count and reset disclosure
**What:** "best of 10 trials", "we exclude runs where the grasp slipped", "the operator
resets the object between trials". Success rates reported over small n, with manual resets,
are signatures of a technique that works only under skilled supervision.
**How:** extract trial counts, exclusion criteria, and human-in-the-loop statements from
experimental protocol text. Indicator = presence/rate, plus disclosed n.

---

## 4. Substitution channel — what carries what text cannot

When authors reach for a non-textual vehicle, they are telling us the text was not
sufficient. The rise of each vehicle is a dated, corpus-wide observable.

### S1 — Carrier portfolio
For every paper, presence and type of: supplementary **video**; **code** release (and its
completeness: full pipeline vs. partial vs. "available on request"); **data/model
weights**; **CAD/BOM/fabrication** files; **benchmark** or challenge participation;
associated **tutorial/workshop**; **standard platform** use.

Video deserves emphasis: it is *demonstrative, not propositional*. It shows without
telling and cannot be reasoned from. Robotics adopted it early and near-universally. The
video-attachment rate, and more interestingly *what the video is doing* (proving a claim
that the text cannot support vs. illustrating one it can), is a proxy for irreducibly
demonstrative content. Classifying video *function* from caption + referring text is a
tractable and, as far as I know, novel measure.

### S2 — Platform concentration
**What:** the fraction of a subfield's empirical work running on a small number of standard
platforms (PR2, TurtleBot, Baxter, Franka Panda, ANYmal, Unitree, Shadow Hand, and in
simulation MuJoCo / Gazebo / Isaac / PyBullet).
**Why it matters:** a standard platform is *material codification* — it substitutes a
purchasable object for MTK that would otherwise have to be rebuilt. Mokyr's standardized
instruments, exactly. Platform adoption should predict a drop in transfer-channel
tacitness, and that prediction is testable.

### S3 — Infrastructure dependency depth
**What:** reliance on shared software infrastructure (ROS, MoveIt, OMPL, PCL, OpenCV, g2o,
Ceres, PyTorch, Isaac Lab) as a count and as a fraction of the pipeline.
**Why:** each dependency is knowledge someone else codified so this author didn't have to.
Infrastructure depth is a cumulative measure of the field's codified stock. It also
predicts the *fragility* story: a field standing on a deep stack it does not understand has
wide effective λ and a hollow Ω.

### S4 — Codification-event registry
**What:** a dated catalogue of discrete codification events — major code releases, dataset
releases, benchmark launches, platform launches, textbook and survey publications, standard
ratifications, tutorial series.
**Why:** these are the treatments in every quasi-experimental design in
[`04`](04-study-design.md). Building this registry carefully is worth as much as any
modeling effort in the project. Assemble semi-automatically (repo creation dates, dataset
DOIs, first-release tags) and curate by hand for the ~200 largest events.

---

## 5. Underdetermination channel — the Executable Specification Probe (ESP)

The most direct question one can ask of a method section is: **could a competent reader
build this from the text alone?** Until recently that question was answerable only by
actually trying, one paper at a time — which is why Raff's 255-paper study is the largest
of its kind. Language models make it answerable at corpus scale, and this is the
methodological centerpiece of the project.

The probe has four variants, in increasing order of cost and decreasing order of
scalability.

### ESP-A — Specification-gap enumeration
Give a model the method section with results, related work, and every code/data reference
stripped. Ask it to produce the list of decisions it would have to make, or questions it
would have to ask the authors, in order to implement the technique. Output: a count and a
taxonomy of gaps (missing hyperparameter, undefined procedure, ambiguous architecture,
unstated preprocessing, unstated hardware setup, unstated tolerance/timing, unstated
failure handling).
**Scales to the whole corpus.**

### ESP-B — Divergence under independent reading *(the key measure)*
Sample *N* independent completions in which the model is asked to fill in every
underdetermined detail and emit a structured specification. Measure the **divergence across
completions**: disagreement on discrete choices, dispersion on continuous ones, structural
edit distance between pipelines.

This measure is the one I would build first, because:

- It requires **no ground truth whatsoever**. It measures a property of the text — how far
  the text constrains a reader — which is exactly the underdetermination we are after.
- It has a clean interpretation as conditional entropy: *H*(implementation | text). This is
  a formally respectable definition of codification deficit, not a proxy for one.
- It is comparable across eras and subfields in a way that lexical markers are not.
- It is cheap: *N* ≈ 8–16 completions per paper on a mid-sized model.

### ESP-C — Ground-truth reconstruction
For papers with an official code release, have the model produce an implementation from the
text alone, then diff structurally against the released code (call graph, module set,
hyperparameter values, pipeline order). What the model gets wrong *is* what the paper failed
to codify. Yields a labeled dataset for calibrating ESP-A/B.

### ESP-D — Parameter recovery rate
The scalar, cheap version of ESP-C: ask the model to predict specific hyperparameters, and
score recovery against released configs. Simple, interpretable, easy to report.

### Contamination control — non-negotiable

A model that has memorized the paper is not measuring the text; it is measuring its own
prior. Mandatory controls:

1. **Temporal holdout.** Reserve papers published after the probe model's training cutoff
   as an uncontaminated stratum. Compare indicator distributions in-cutoff vs. out.
2. **Redaction.** Strip title, author names, affiliations, method names, distinctive
   dataset names, and characteristic numbers before probing. Verify redaction adequacy by
   asking the model to name the paper.
3. **Recognition probe.** Explicitly measure recognition rate per paper and use it as a
   covariate — or exclude recognized papers and report the sensitivity.
4. **Model triangulation.** Run ESP with ≥ 2 model families with different training data;
   report cross-model correlation. If the measure is a property of the text rather than of
   any one model, models should agree.
5. **Counterfactual paraphrase.** Rewrite a sample of method sections to be *equally
   detailed but lexically different*; the measure should be stable. Rewrite them to be
   *less* specific (delete a known parameter); the measure should move in the predicted
   direction. This is the strongest single check that ESP measures specification content
   rather than surface form — see [`05`](05-validation-and-threats.md) §2.

### An honest limit
ESP measures whether the text determines an implementation *for a reader who already has
the field's background*. It therefore captures RTK well, MTK partially (it will flag
"unstated hardware setup"), and STK/CTK barely at all — the model has no body and no
community membership. State this plainly rather than letting the measure over-claim.

---

## 6. Epistemic-base measurement (Mokyr's B)

Detecting the deficit is only half the study; the framework's real question is how the
deficit relates to the propositional knowledge underneath. Per technique per period:

### B1 — Justification-type classification
Label each technique's core "why it works" claim: **derived** (from a stated model, with
argument) / **bounded** (formal guarantee) / **analogical** / **bio-inspired** /
**empirical-only** / **absent**. Sentence-level classification over the passages where the
technique is introduced or defended.

### B2 — Reference reach and depth
Fraction of supporting citations outside robotics venues (into mathematics, control theory,
physics, statistics, neuroscience); and the age distribution of those citations. A wide,
deep base cites old and external. A narrow base cites recent and internal.

### B3 — Explanatory vocabulary ratio
Density of model-presupposing terms (stability, convergence, observability, identifiability,
bound, guarantee, invariant, optimality, consistency) relative to purely performative ones
(achieves, outperforms, demonstrates, shows, state-of-the-art). Blunt but cheap, and a good
sanity check on B1.

### B4 — Stated operating envelope
Does the paper state conditions under which the technique fails or does not apply? Papers
with a real epistemic base can draw their own boundary; papers without one cannot. Note the
productive tension with R1's caveat class — the same sentence can indicate *both* a
codification effort and a real base. That overlap is a feature: it is where Ω and λ meet,
and those sentences deserve their own qualitative reading.

---

## 7. Composing the indicators

Do **not** average these into one number. Instead:

1. **Estimate a latent factor model** over the indicator battery, per technique-year, with
   channels as separate measurement blocks. Test whether the indicators load on one factor
   (a general codification deficit) or several (which would be the more interesting and, I
   expect, the correct result — plausibly a *specification* factor from ESP + R, a
   *transfer* factor from T, and an *embodiment* factor from F3 + S1 + S2).
2. **Report each channel separately** in all headline results. If the transfer channel and
   the linguistic channel disagree about a subfield, that disagreement is a finding, not
   noise.
3. **Weight by measurement quality.** The transfer and failure channels are behavioral and
   robust; the repair channel is linguistic and fragile. Let the model know that.
4. **Preserve the Collins taxonomy.** Tag every indicator with the tacit kind(s) it can
   detect, and never claim STK/CTK evidence from an RTK-only instrument.

## 8. Build order

Priority for the first pass, by (value × feasibility) ÷ cost:

| Tier | Indicators | Rationale |
|---|---|---|
| **1 — build first** | ESP-B, T1, T2, S1, S4, F1 | Metadata-only or ground-truth-free; highest leverage; ESP-B is the novel instrument and F1 gives a validation set almost for free. |
| **2** | R1, R3, B1, B2, F2, F3, S2, S3 | Need full text and/or annotation rounds. |
| **3** | T3, T4, R2, R4, B4, ESP-C/D | Need author disambiguation, structural parsing, or code-linked subsets. |
