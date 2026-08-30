# Appendix — Deferred components

Components designed but explicitly **out of scope for the current study**. Kept on record
because each has a clear re-entry condition, and because a reader of the plan should be
able to see what was considered and why it was set aside.

---

## A. Expert elicitation survey — *deferred*

**What it was.** A survey of 30–60 active roboticists rating a stratified sample of ~50
technique-year pairs: *if you had only this paper, could your lab reproduce this? What
would you need to ask the authors? Would you expect to need to visit, or to hire someone?*

**Why it mattered.** It was the design's primary **construct-validity anchor** — the only
line of validation connecting our indicators to practitioner judgment about the actual
construct. Its removal is the single largest weakening of the design, and
[`05`](05-validation-and-threats.md) §V0 states honestly what has to carry the load
instead.

**What is lost with it, specifically:**

1. **Direct construct validity.** We can no longer show that an indicator correlates with
   what a roboticist means by "you'd have to visit their lab to make this work". Everything
   that remains is either predictive validity (V3), face validity (V1), or internal
   consistency (V5).
2. **The insider/outsider CTK sub-study.** Asking respondents about techniques inside and
   outside their own subfield, and measuring the gap, was the study's only real instrument
   on **collective tacit knowledge**. Without it, CTK is reachable only obliquely through
   U2's unnamed-convention class ("standard practice", "the usual approach"), which is a
   lexical hook, not a measurement. **CTK is effectively out of scope, and the papers
   should say so rather than gesture at it.**
3. **A calibration scale.** Expert ratings would have given the composite indicators
   meaningful units. Without them, all results are relative — rankings, gradients, and
   event-study deltas, never levels. This is survivable (every hypothesis in
   [`04`](04-study-design.md) §4 is comparative) but it should be a conscious constraint on
   how results are phrased. **No claim of the form "robotics is X% tacit."**

**Re-entry condition.** Before the findings paper (Paper 2), if at all possible. IRB lead
time is the binding constraint, so if the survey is ever likely to happen, filing early
costs little and buys the option. A reduced version — 15 experts, 20 items, validation
only, no CTK sub-study — would recover most of line (1) at a fraction of the effort and is
worth reconsidering once the indicators exist and there is something concrete to show
respondents.

---

## B. Model-in-the-loop underdetermination probe (the "ESP") — *deferred, possibly permanently*

**What it was.** Four variants of a probe that asks whether a competent reader could build
a technique from the text alone:

- **ESP-A** — enumerate every decision the text leaves undetermined.
- **ESP-B** — sample *N* independent reconstructions and measure **divergence across them**,
  interpretable as conditional entropy *H*(implementation | text). Required no ground truth
  at all, which was its appeal.
- **ESP-C** — reconstruct, then diff structurally against the official code release.
- **ESP-D** — predict specific hyperparameters; score recovery against released configs.

**Why it is deferred.** See [`02`](02-detection-methods.md) §0. In short: instrument
constancy (a forty-year trend measured with a moving instrument is not a trend),
contamination (the model has read the later corpus and the very repositories that are
supposed to be the outcome), and auditability. The contamination problem is the decisive
one, because it is *correlated with the time axis* — models have memorized recent papers
far better than 1990s ones, so an ESP time series would show a spurious trend in exactly
the direction the study is trying to test. No weighting scheme fixes that cleanly.

**What replaces it.** [`02`](02-detection-methods.md) §5: specification-checklist
completeness (U1), delegation density and chain depth (U2), and hedge-attachment (U3).
These measure whether the text *contains* the specification; ESP measured whether the
specification was *sufficient*. The substitution is a genuine downgrade in construct
sharpness, and U1's write-up concedes it.

**Re-entry conditions**, all of which would have to hold:

1. A **frozen, versioned, self-hosted** model, archived alongside the paper so the
   measurement is re-runnable.
2. A **training-cutoff-based contamination design** — the probe applied only to papers
   published after the model's cutoff, with the pre-cutoff corpus used solely as a
   contamination control rather than as data.
3. Passing **ablation and enrichment** tests: deleting a stated specification element must
   raise the measure monotonically; splicing one in from released code must lower it. If
   those fail, the probe is measuring prose, not specification, and should stay retired.
4. Paraphrase invariance and cross-model agreement.

Even then it should enter as a **secondary, cross-sectional** measure on a recent
sub-corpus — never as a component of a long-run time series. That restriction is not
negotiable and is the reason "possibly permanently".
