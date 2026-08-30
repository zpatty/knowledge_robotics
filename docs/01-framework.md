# 01 — Conceptual framework

## 1.1 Mokyr's two knowledges

*The Gifts of Athena* (2002) partitions useful knowledge into two sets:

- **Ω (propositional knowledge)** — beliefs about natural regularities. "What is." Rigid-body
  dynamics. The geometry of epipolar constraints. The convergence conditions of an
  extended Kalman filter. Contact mechanics.
- **λ (prescriptive knowledge)** — techniques, instruction sets, recipes. "What to do."
  Compute the Jacobian pseudo-inverse and damp it near singularities. Randomize friction
  coefficients between 0.4 and 1.2 during training. Tighten the tendon until it just
  stops slipping.

Every technique λ rests on an **epistemic base**: the subset of Ω that explains *why* it
works. Mokyr's central historical claim is that the *width* of that base governs a
technique's fate. A technique with a narrow base can be executed and copied but not
reliably extended, debugged, or taught at scale — improvement proceeds by trial. A
technique with a wide base can be reasoned about, generalized, and transmitted cheaply,
which is what turns isolated invention into sustained growth. His second claim is that
falling **access costs** to Ω — journals, societies, encyclopedias, standardized
instruments, shared notation — are the mechanism by which the base widens.

Robotics is an unusually good test bed for this framework because it spans, inside one
publication venue, the full range from techniques with essentially complete epistemic
bases (linear control of a modeled plant) to techniques with almost none (reward shaping
for a manipulation policy; laying up a silicone actuator so it does not delaminate).

## 1.2 Where tacit knowledge sits

Tacit knowledge is the part of λ that does not fit into the instruction set. Polanyi's
formulation — "we can know more than we can tell" — is the right starting point, but for
a measurement project we need Collins' (*Tacit and Explicit Knowledge*, 2010) sharper
three-way split, because the three kinds have very different observable consequences and
very different policy implications:

| Kind | Definition | Robotics example | Is codification possible? |
|---|---|---|---|
| **Relational (RTK)** | Could in principle be written down; isn't, for contingent reasons — page limits, it seemed obvious, nobody asked, competitive advantage. | The actual gain schedule; the fact that the demo only works after the motors warm up. | Yes. Most of what we will detect. |
| **Somatic (STK)** | Resides in a body or a machine's material particulars; limited by embodiment. | Hand-tuning an impedance controller by feel; hand-laying composite; "knowing" a gripper is about to slip. | Partially — can be replaced by different techniques, not transcribed. |
| **Collective (CTK)** | Resides in a community's shared judgment; depends on socialization. | What counts as a fair baseline comparison; what "works reliably" means; which failure modes are worth reporting. | No, not into text. **Effectively out of scope** — see [`appendix-deferred.md`](appendix-deferred.md) §A. |

A fourth category matters specifically in robotics and I will treat it as its own kind:

| **Material (MTK)** | Resides in a particular physical instantiation and the fit between its parts. | This PR2's calibration; the lab's particular lighting; the specific batch of actuators. | Substituted, not codified — by standard platforms. |

**Design consequence:** a single scalar "tacitness score" would be a category error. The
measurement instrument must be multi-dimensional, and each indicator must be labelled
with which kind(s) it plausibly detects. A field can become less RTK-laden while its
STK/MTK floor is unchanged, and those are different stories about the field.

## 1.3 The construct we actually measure

Direct measurement of tacit knowledge is impossible by construction. What is measurable
is the **codification deficit**:

> **D(λ, t)** = (what a practitioner must know to execute technique λ successfully at
> time t) − (what is recoverable from the public written record at time t)

Neither term is directly observable, but *changes* and *comparisons* in D are, through
the five channels in [`02-detection-methods.md`](02-detection-methods.md). Three framing
points:

1. **We measure a deficit, not an essence.** D includes both irreducibly tacit knowledge
   and merely-uncodified knowledge. Separating them is an inference problem, not a
   measurement problem, and the honest position is that our indicators primarily track
   RTK + MTK, reach STK only obliquely, and mostly cannot see CTK at all. Say so in the paper.
2. **Codification is an event, not a state.** We cannot see tacit knowledge, but we *can*
   see it disappearing: a code release, a benchmark, a standard platform, a tutorial, a
   textbook chapter. Codification events are dated, discrete, and abundant in robotics.
   Much of the empirical leverage comes from studying the field around them.
3. **The interesting quantity is relational, and probably cyclical.** The question the
   study is really asking is not "how tacit is robotics" but *how D and the epistemic base
   co-evolve*. Mokyr's history contains both directions — practice ahead of theory, theory
   ahead of practice — and the working hypothesis here is that these are phases of one
   oscillation rather than rival regimes: practice runs ahead at a frontier, theory catches
   up and widens the base, the widened base makes codification cheap, codification opens a
   new frontier. See [`04`](04-study-design.md) §4, H5, where this is developed and where
   the choice of statistics follows from it.

4. **We measure comparatively, never absolutely.** With the expert survey deferred
   ([`appendix-deferred.md`](appendix-deferred.md) §A) the indicators have no calibrated
   units. Every claim is a ranking, a gradient, an event-study delta, or a phase
   relationship. No statement of the form "robotics is X% tacit" is supportable, and none
   should appear.

## 1.4 The second construct: epistemic base width

For each technique λ we also need **B(λ, t)**, the width of its epistemic base. Working
operationalization (details in [`02`](02-detection-methods.md), §7):

- **Justification type** of the technique's core claims: derived-from-model / bounded /
  analogical / biologically-inspired / empirical-only / unjustified.
- **Reference reach**: fraction of the technique's cited support that lies outside the
  robotics literature (mathematics, physics, statistics, control theory, neuroscience),
  and the age distribution of those citations — a deep base cites old, stable results.
- **Explanatory vocabulary density**: presence of terms that presuppose a model
  (stability, convergence, observability, consistency, bound, guarantee, optimality) versus
  purely performative ones (works, achieves, outperforms, demonstrates).
- **Generalization statements**: does the text state conditions under which the technique
  will and will not work? A stated failure boundary is strong evidence of a real base.

## 1.5 Why this is not just bibliometrics

Two commitments distinguish this from a standard scientometrics paper:

- **The unit of analysis is the technique, not the paper.** Knowledge lives in λ-objects
  (ICP, MPPI, domain randomization, RRT*, impedance control, tendon routing, cable-driven
  transmission design) that persist across hundreds of papers and decades. Papers are
  observations of techniques, not the things themselves. See [`04`](04-study-design.md) §1.
- **The instruments are auditable by construction.** Every measure is built from lexicons,
  rule systems, parses, and supervised classifiers trained on published annotation
  guidelines — no measure depends on a language model's judgment of a text. The reasoning
  is in [`02`](02-detection-methods.md) §0, and the short version is that a forty-year
  trend cannot be measured with an instrument whose calibration cannot be held fixed, and
  whose training data includes the later half of the corpus.

## 1.6 Prior work this builds on

- Mokyr, *The Gifts of Athena* (2002); Ω/λ, epistemic base, access costs.
- Polanyi, *Personal Knowledge* (1958), *The Tacit Dimension* (1966).
- Collins, *Changing Order* (1985) — the TEA-laser case, in which no lab succeeded in
  building the laser from published descriptions alone without personal contact. This is
  the canonical demonstration of the transfer channel and the direct ancestor of §3 of
  [`02`](02-detection-methods.md).
- Collins, *Tacit and Explicit Knowledge* (2010) — RTK/STK/CTK.
- MacKenzie & Spinardi, "Tacit Knowledge, Weapons Design, and the Uninvention of Nuclear
  Weapons," *AJS* (1995) — knowledge loss when the tacit substrate is not reproduced.
- Nelson & Winter, *An Evolutionary Theory of Economic Change* (1982) — routines as the
  locus of tacit skill in organizations.
- Vincenti, *What Engineers Know and How They Know It* (1990) — engineering knowledge
  categories; the closest disciplinary precedent for what we are attempting.
- Cowan, David & Foray, "The Explicit Economics of Knowledge Codification and Tacitness,"
  *Industrial and Corporate Change* (2000) — argues much "tacit" knowledge is simply
  uncodified because codification wasn't worth the cost. This is the sceptical position
  our RTK/STK separation has to take seriously.
- Latour & Woolgar, *Laboratory Life* (1979); Knorr Cetina, *Epistemic Cultures* (1999) —
  inscription practices and the material grounding of laboratory knowledge.
- Raff, "A Step Toward Quantifying Independently Reproducible Machine Learning Research,"
  NeurIPS (2019) — one author's attempts to reproduce 255 papers, regressed on paper
  features. The nearest existing quantitative work and a validation target for us.
- Robotics-specific reproducibility literature (Bonsignorio, del Pobil and colleagues in
  *IEEE Robotics & Automation Magazine*; the RA-L reproducible-article track). *Verify
  exact citations and the current status of the R-Article program before writing.*
