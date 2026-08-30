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

## 0. Instrument constraint: no LLM-based text assessment

**Every indicator in this document is computable with lexicons, rule systems, dependency
parsing, structural document parsing, or supervised classifiers trained on human-labeled
data.** No indicator depends on a language model's judgment of a text. This is a binding
constraint on the design, not a preference, and there are three reasons it is the right
one:

1. **Instrument constancy.** The study's central claims are about change across forty
   years. You cannot measure a forty-year trend with an instrument whose calibration you
   cannot hold fixed. Model versions are deprecated, retrained, and silently updated; a
   result produced by one is not reproducible by the next. A lexicon and a rule set are
   fixed objects that a reader can inspect and re-run in ten years.
2. **Contamination.** Any model of recent vintage has trained on the later part of our
   corpus and on the code repositories that are supposed to be our *outcome*. An
   LLM-derived measure of "how underspecified is this 2023 paper" is partly a measure of
   how well the model memorized that paper, and the contamination is worst exactly where
   the corpus is densest. There is no clean way to correct for this in a time-series
   design.
3. **Auditability.** A reviewer can check a regex, a checklist, and an annotation
   guideline. They cannot check a model's judgment, and a study whose headline construct
   is unauditable will not — and should not — persuade.

The cost is real and worth stating plainly: the underdetermination channel (§5) is
sharper when a reader actually attempts reconstruction than when we count specification
elements. §5 is designed around that loss. A model-based probe is documented as deferred
future work in [`appendix-deferred.md`](appendix-deferred.md), to be revisited only with
a frozen, versioned, self-hosted model and an explicit contamination analysis.

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

**Build:** a seed lexicon (~300 patterns) for recall, plus syntactic constraints (the
necessity-without-reason class needs a parse to establish the *absence* of a justifying
subordinate clause — that is a dependency-parse test, not a keyword test). Hand-label a
stratified sample of 3–5k sentences against a written annotation guideline; train a small
supervised encoder (DeBERTa-v3 / ModernBERT-scale) on those labels. Report per-class κ
against two independent human annotators and release the guideline with the paper.
**Cost:** moderate — one substantial annotation round. Inference is cheap.
**Confounds:** *This is the most confounded indicator in the battery.* Writing style,
venue norms, page limits, and non-native-English authorship all move it. It must never be
used alone, must be normalized within year × subfield × page-count strata, and must be
reported alongside the style placebo (see [`05`](05-validation-and-threats.md) §V4).

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
Normalize by method-section length. Rule-based with a hand-audited sample.
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
it needs no annotation, and it is computable from metadata alone (so it works even where
full text is unavailable). It is also entirely unaffected by the §0 constraint.

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
where recoverable (thesis metadata, genealogy projects, lab web pages). Personnel movement
is inferable from affiliation change across a person's publication sequence.
**Note:** T1 and T2 are the pair that lets us make the study's cleanest causal statement.
See the diff-in-diff in [`04`](04-study-design.md) §4, H2.

### T3 — Citation/implementation divergence
**What:** techniques cited far more often than they are *used*. A method everyone cites and
nobody reimplements is a method whose paper does not contain it.
**How:** classify each citation context into: background mention / comparison baseline /
*actual use* / extension. Citation-intent classification is a well-established supervised
task (SciCite-style label schemes and models); fine-tune on ~2k hand-labeled
robotics-specific contexts. Indicator = use-citations / total citations, with
baseline-comparison citations tracked separately (being used as a baseline usually implies
a working reimplementation *or* a released one — which of the two is itself informative).
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
**Cost:** low to extract, and disproportionately valuable — with the expert survey tabled,
F1 is now one of the few sources of near-ground-truth in the design. **Prioritize.**

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
telling and cannot be reasoned from. Robotics adopted it early and near-universally. If
IEEE's metadata carries a multimedia/supplementary flag ([`03`](03-corpus.md) §3.4), we
get video presence for the *entire* corpus, unbiased, across four decades — which would be
a first-class descriptive result on its own. Classifying video *function* from caption and
referring text (proving a claim the text cannot support vs. illustrating one it can) is a
further, tractable measure.

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
[`04`](04-study-design.md), *and* — with the survey tabled — the dose-response response of
our indicators to these events is now the study's primary validation anchor
([`05`](05-validation-and-threats.md) §V3). Building this registry carefully is therefore
worth as much as any modeling effort in the project. Assemble semi-automatically (repo
creation dates, dataset DOIs, first-release tags) and curate by hand for the ~200 largest
events.

---

## 5. Underdetermination channel — without a model in the loop

The question is unchanged: **could a competent reader build this from the text alone?**
The honest way to answer it is to have someone try, which is why Raff's 255-paper
reproduction study is the largest of its kind and why nobody has done it at corpus scale.
Under the §0 constraint we cannot simulate that reader. What we *can* do is decompose the
question into three transparent, auditable measures that together approximate it.

### U1 — Specification-checklist completeness *(the workhorse)*
**What:** for each method family, a hand-authored checklist of the specification elements a
working implementation actually requires. Then: what fraction does the paper supply?

Examples of family checklists (~15 families, 12–25 items each):

| Family | Required elements (abbreviated) |
|---|---|
| Model-based control | plant model, state definition, gains, sample rate, saturation limits, initialization, tuning procedure, stability conditions |
| SLAM / estimation | sensor model, noise parameters, initialization, loop-closure criteria, outlier rejection thresholds, map representation, timing |
| Learned policy | observation space, action space, reward function, architecture, optimizer + LR, episode length, termination conditions, randomization ranges, training budget, eval protocol, seeds |
| Manipulation hardware | gripper spec, object set, pose initialization, tolerance, calibration procedure, reset procedure, failure criteria |
| Fabrication | materials + supplier, cure/deposition parameters, mold geometry, post-processing, yield rate |

**How:** rule-based extraction per checklist item, keyed to the family classifier. Score =
fraction present. Crucially, encode the *adequacy* distinction in the rules: "learning
rate: 3e-4" is present; "we tune the learning rate" is not. Hand-audit a stratified sample
of 300 papers to measure extraction precision per item.
**Cost:** high up-front — checklist authoring genuinely requires the domain roboticist, and
this is where their time is best spent. Cheap thereafter, and fully reusable.
**Precedent:** this is the NeurIPS/ML reproducibility checklist idea, except computed from
the text rather than self-reported by authors — which removes the honesty problem and makes
it retroactively applicable to 1984.
**Limitation, stated up front:** presence is not sufficiency. A paper can name every
checklist item and still be unbuildable. U1 measures a necessary condition, not the
construct itself. This is the real cost of dropping the model-in-the-loop probe, and it
should be conceded in the paper rather than argued around.

### U2 — Delegation density and chain depth
**What:** the places where a paper hands its specification off somewhere else — "as
described in [12]", "we use standard settings", "following [X]", "default parameters", "as
in our previous work", "the usual approach". Each is a point where the text is not
self-contained.
**How:** lexical patterns plus reference resolution. Classify each delegation by *target*:

- **to a citation** — transitively recoverable. Follow the chain: how many hops to a
  self-contained specification, and does the chain terminate at one, or in a loop, or at a
  paper nobody can access? **Delegation-chain depth is a genuinely novel measure and it is
  cheap.** A field whose specifications bottom out in three hops is differently codified
  from one whose chains run to eight.
- **to a code artifact** — codified, and dated (feeds S4).
- **to an unnamed convention** — "standard practice", "the usual", "common settings". This
  is *collective tacit knowledge made visible in text*, and it is the only lexical hook we
  have on CTK. With the insider/outsider survey tabled, this is now the study's sole CTK
  instrument, which makes it more important than its simplicity suggests.
- **to nothing** — dangling.

**Cost:** low. **Confounds:** page limits drive delegation upward independent of
tacitness — normalize within venue-year, and note that this cuts the *opposite* way from
the usual story (a tight page limit forces codification out of the text without changing
what is known).

### U3 — Hedge attachment to specification
**What:** hedging *attached to a specification element* — "approximately 0.5", "a suitable
gain", "an appropriate threshold", "carefully chosen", "roughly 20 trials" — as distinct
from hedging in general, which is a style measure and useless here.
**How:** dependency parse; count hedge terms only where they modify a parameter, threshold,
quantity, or procedure. The attachment requirement is what keeps this from collapsing into
a readability score. Base the hedge lexicon on Hyland's taxonomy, restricted by attachment.
**Cost:** low-moderate. Needs a parser that survives 1980s two-column OCR — check this
early on old papers, since parse quality is likely to degrade going back in time and that
degradation would masquerade as a time trend.

### What U1+U2+U3 do and don't buy
Together they estimate whether the text *contains* the specification. They do not estimate
whether the specification is *sufficient* — that requires an attempted reconstruction. Treat
the composite as a lower bound on the codification deficit, report it as such, and rely on
the transfer and failure channels for claims that need the stronger construct.

---

## 6. Epistemic-base measurement (Mokyr's B)

Detecting the deficit is only half the study; the framework's real question is how the
deficit relates to the propositional knowledge underneath. Per technique per period:

### B1 — Justification-type classification
Label each technique's core "why it works" claim: **derived** (from a stated model, with
argument) / **bounded** (formal guarantee) / **analogical** / **bio-inspired** /
**empirical-only** / **absent**. Supervised sentence classification over the passages where
the technique is introduced or defended, on the same annotation footing as R1.

### B2 — Reference reach and depth
Fraction of supporting citations outside robotics venues (into mathematics, control theory,
physics, statistics, neuroscience); and the age distribution of those citations. A wide,
deep base cites old and external. A narrow base cites recent and internal. **Computable
from metadata alone** — which makes B2 the one epistemic-base indicator available for the
Phase 1 metadata-only study, and therefore the one that carries H5 in its first pass.

### B3 — Explanatory vocabulary ratio
Density of model-presupposing terms (stability, convergence, observability, identifiability,
bound, guarantee, invariant, optimality, consistency) relative to purely performative ones
(achieves, outperforms, demonstrates, shows, state-of-the-art). Blunt but cheap, works on
abstracts alone, and a good sanity check on B1.

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
   expect, the correct result — plausibly a *specification* factor from U + R, a
   *transfer* factor from T, and an *embodiment* factor from F3 + S1 + S2).
2. **Report each channel separately** in all headline results. If the transfer channel and
   the linguistic channel disagree about a subfield, that disagreement is a finding, not
   noise. With no external human anchor, **inter-channel convergence is now carrying real
   validation weight** ([`05`](05-validation-and-threats.md) §V5), so the channels must be
   built from disjoint data sources and kept that way.
3. **Weight by measurement quality.** The transfer and failure channels are behavioral and
   robust; the repair channel is linguistic and fragile. Let the model know that.
4. **Preserve the Collins taxonomy.** Tag every indicator with the tacit kind(s) it can
   detect, and never claim STK/CTK evidence from an RTK-only instrument. Under the current
   scope, CTK is reachable only through U2's unnamed-convention class, and that should be
   said out loud rather than quietly over-read.

## 8. Build order

Priority for the first pass, by (value × feasibility) ÷ cost:

| Tier | Indicators | Rationale |
|---|---|---|
| **1 — build first** | T1, T2, S1, S2, S4, B2 | Metadata-only. Immune to the full-text access risk, immune to the §0 constraint, and sufficient for the Phase 1 study and a first pass at H2 and H5. |
| **2** | F1, U1, U2, R1, R3, B1, B3 | Need full text (or abstracts, for B3). F1 first within this tier — it is cheap and it is ground truth. U1 gated on checklist authoring. |
| **3** | T3, T4, R2, R4, F2, F3, F4, U3, B4 | Need author disambiguation, structural parsing, dependency parsing, or table extraction. |
