# Analysis, restructured — plan and today's results (2026-09-08)

## Why restructure

The section is currently built on "the accuracy lives in the feature and the logit route loses it".
Today's cells do not support that:

* frozen-head logit MSE ties the feature anchor on both datasets (CIFAR-100 NRR 36.43 vs 36.64;
  CIFAR-10 47.93 vs 47.90 AutoAttack, every metric inside two decimals);
* CIFAR-10 makes the tie sharper, not weaker, even though a logit target there carries 9 numbers per
  example against the feature's 512;
* ARREST already distils a representation from a frozen natural teacher, so the layer choice is not
  ours to claim in the first place.

What the cells do support is a different section: **what a natural teacher transfers is class
geometry, and the knobs that matter are the ones that decide how much of it survives** — the
teacher's own training length, whether the target passes through a softmax, whether the label loss is
kept, and what the attack ascends. The counterintuitive result (a better-trained teacher makes a
worse student) becomes the spine instead of an appendix remark.

## Proposed structure

| § | content | status of the evidence |
|---|---|---|
| 2.1 | preliminaries, teacher notation | unchanged |
| 2.2 | **teacher training length -> class geometry -> student.** The ladder, r = -0.999, geometry monotone where accuracy is not | measured; weak-teacher rungs queued |
| 2.3 | **which target.** temperature (both sides vs teacher only), MSE, feature. Geometry of each student | measured today |
| 2.4 | **where the teacher is read**, and **what the attack ascends** | measured (0.00 AA control; CE-attack block) |
| 2.5 | **what the label loss costs** | measured (lambda sweep) |
| 3 | method = the configuration those four select, plus sensitivity-matched epsilon | unchanged |

## Today's results

### 1. Targets, CIFAR-100, 200-epoch teacher (50 epochs, eps 8/255, no stack)

| target | mean max prob | entropy (nats) | Clean | AA | NRR |
|---|---|---|---|---|---|
| teacher-only KD, tau=1 | 0.820 | 0.76 | 58.26 | 20.84 | 30.70 |
| teacher-only KD, tau=2 | 0.533 | 2.40 | 59.33 | 22.79 | 32.93 |
| teacher-only KD, tau=4 | 0.154 | 4.20 | 59.39 | 24.48 | 34.67 |
| teacher-only KD, tau=8 | 0.040 | 4.56 | 59.47 | 25.19 | 35.39 |
| teacher-only KD, tau=16 | 0.020 | 4.60 | 57.78 | 24.00 | 33.91 |
| conventional KD, tau=2 | | | 58.90 | 23.02 | 33.10 |
| conventional KD, tau=4 | | | 60.83 | 25.05 | 35.49 |
| conventional KD, tau=8 | | | 60.99 | 25.25 | 35.71 |
| conventional KD, tau=16 | | | 61.85 | 25.39 | 36.00 |
| conventional KD, tau=32 | | | 61.81 | 25.50 | 36.11 |
| frozen-head logit MSE, lr 0.007 | --- | --- | 62.60 | 24.43 | 35.14 |
| **frozen-head logit MSE, lr 0.021** | --- | --- | 62.59 | 25.69 | **36.43** |
| **feature anchor** | --- | --- | 62.72 | 25.88 | **36.64** |

Entropy is $\mathbb{E}_x[H(\mathrm{softmax}(f_t(x)/\tau))]$ on clean test inputs; $\log C = 4.61$.

### 2. Class geometry of those students ($S_w/S_b$, lower = better separated)

| target | clean | under attack |
|---|---|---|
| KD tau=1 | 4.337 | 4.982 |
| KD tau=2 | 3.406 | 4.644 |
| KD tau=4 | 1.796 | 3.400 |
| KD tau=8 | 1.434 | 3.065 |
| KD tau=16 | 1.292 | 2.746 |
| **feature anchor** | **0.850** | **2.190** |
| natural teacher | 0.808 | 8.179 |

Softening moves the geometry monotonically and never recovers the clean accuracy. The anchor lands on
the teacher's clean geometry (0.850 against 0.808) without inheriting its collapse under attack
(2.190 against 8.179).

### 3. Shared attack: same objectives, one true-label CE-PGD

| target | own attack | CE attack | dClean | dAA |
|---|---|---|---|---|
| feature anchor | 62.72 / 25.88 (36.64) | 63.47 / 25.27 (36.15) | +0.75 | -0.61 |
| frozen-head logit MSE | 62.59 / 25.69 (36.43) | 63.15 / 24.78 (35.59) | +0.56 | -0.91 |
| conventional KD tau=16 | 61.85 / 25.39 (36.00) | 62.27 / 24.65 (35.32) | +0.42 | -0.74 |
| teacher-only KD tau=4 | 59.39 / 24.48 (34.67) | 60.47 / 24.23 (34.60) | +1.08 | -0.25 |

The ordering is the same under both attacks, so the difference between objectives is not produced by their
attacks. ARREST's choice (representation loss, CE attack) is the second row of the anchor line.

### 4. CIFAR-10, 200-epoch teacher, own attacks

| target | Clean | PGD-20 | CW | AA | NRR |
|---|---|---|---|---|---|
| feature anchor | 85.00 | 49.75 | 50.40 | 47.90 | 61.27 |
| frozen-head logit MSE | 85.28 | 49.73 | 50.39 | 47.93 | 61.37 |
| conventional KD tau=16 | 84.97 | --- | --- | 47.65 | 61.06 |
| teacher-only KD tau=4 | 84.88 | --- | --- | 47.07 | 60.56 |

### 5. The same four targets on the 50-epoch teacher (CIFAR-100)

| target | teacher 200ep | teacher 50ep |
|---|---|---|
| feature anchor | 62.72 / 25.88 (36.64) | **64.28** / 24.38 (35.35) |
| frozen-head logit MSE | 62.59 / 25.69 (36.43) | 63.06 / 24.24 (35.02) |
| conventional KD tau=16 | 61.85 / 25.39 (36.00) | 62.84 / 24.73 (**35.49**) |
| teacher-only KD tau=4 | 59.39 / 24.48 (34.67) | 61.27 / **24.77** (35.28) |

The ranking is not a property of the target alone. On the weaker teacher the four collapse into a
1.5-point band and the softened-logit target is best on AutoAttack, while the anchor keeps the clean
advantage and grows it (+0.13 to +1.22).

### 6. What the label loss costs (8.8/255, 50 epochs, existing cells)

| | Clean | AA | NRR |
|---|---|---|---|
| anchor alone | 61.19 | 26.41 | 36.90 |
| anchor + 0.1 CE | 61.11 | 24.38 | 34.83 |
| anchor + 0.3 CE | 61.04 | 24.12 | 34.60 |
| anchor + 1.0 CE | 60.84 | 23.65 | 34.28 |

### 7. Decomposition, all at the same regime

| step | NRR |
|---|---|
| teacher-only KD, tau=4 | 34.67 |
| -> temperature on both sides (tau=16) | 36.00 (+1.33) |
| -> drop the softmax (logit MSE) | 36.43 (+0.43) |
| -> logits to features | 36.64 (+0.21) |
| separately: adding back 0.1 x label CE | -2.03 AA |

### 8. What the prior methods actually do

| method | teacher | distilled quantity | attack objective | student-side temperature |
|---|---|---|---|---|
| ARREST | natural, frozen | penultimate representation, angular distance | **label CE only** | --- |
| LBGAT | natural, trained jointly | logits, MSE, no softmax | TRADES KL, student only | --- |
| ADR | own EMA | rectified soft label | **the rectified label** | none (label replaces y) |
| B-MTARD | natural + robust | softened logits, entropy-balanced | student PGD | **constant 1** |
| ARD | robust | softened logits | label CE | tau (standard KD, tau^2) |
| RSLAD | robust | soft labels | teacher soft labels | none (tau = 1) |
| AdaAD | robust | logits inside the ball | student-teacher disagreement | none |
| ours | natural, frozen | clean feature, l2 | the anchor itself | --- |

Two of the four temperature-based methods keep the student at a fixed temperature, so the
teacher-only variant is not a straw man; it is ADR's and B-MTARD's setting.

### 9. Gradient scaling, measured on real logit statistics

With the same temperature on both sides the KD gradient decays as $1/\tau^2$ and the $\tau^2$ factor
gives a finite limit equal to mean-removed logit matching (0.00439 for both at $\tau \ge 32$). With
the teacher softened alone the gradient is $O(1)$ in $\tau$ and the target converges to the uniform
distribution (max probability 0.0122 at $\tau = 64$ against $1/C = 0.01$), so raising the temperature
there is not softening the supervision but pushing the student toward uniform. This is why the
teacher-only curve turns over at $\tau = 16$ while the conventional one plateaus.

## Still running or queued

* CIFAR-10 shared-attack block (4 cells)
* logit MSE and conventional KD at 8.8/255 for 100 epochs, no stack — does the tie survive the scale
* the same two at the full stack — does the tie survive weight averaging and AWP
* weak teachers (5/10/20/40 clean epochs) and their students — extends the teacher ladder below 75.8
* table-5 remainder, epoch and learning-rate sensitivity

## Open decisions

1. If the tie survives the stack, the paper's identity moves from "feature over logit" to "label-free
   regression onto a natural teacher, plus the allocation rule", and the layer becomes an ablation.
2. The teacher-quality ladder decides whether we can say "the teacher must be trained to some degree"
   or only "training it further does not help". Today's 50-epoch cells already show the ranking of
   targets depends on the teacher, which is the more interesting half of that claim.
3. Table 1 should report conventional KD, with the teacher-only variant kept as the ADR / B-MTARD
   setting rather than dropped.

## 새로운 의견 (2026-09-09)

### Revised central claim

Feature distillation alone cannot be presented as the main novelty because ARREST already preserves
the representation of a naturally pretrained network during adversarial fine-tuning. The more
substantive distinction is the role assigned to that representation. ARREST retains adversarial
cross-entropy as the primary objective and uses representation matching as a weighted regularizer;
its adversarial examples are also generated using cross-entropy. CFA instead elevates the clean
representation from an auxiliary constraint to the sole target of adversarial training. The same
feature discrepancy is maximized to generate adversarial examples and minimized to train the
backbone, while the teacher's classifier is inherited and kept fixed.

The central message should therefore be:

> Rather than constructing a softened predictive distribution from a natural teacher and balancing
> it against a label loss, CFA uses the teacher's clean representation directly as a fixed target.
> This replaces target calibration and loss balancing with a single feature-space anchor.

Under this formulation, CFA requires neither a distillation temperature nor a coefficient balancing
teacher and label supervision. The proposition $L\le F+O\le3L$ explains why one anchor can serve two
roles: it bounds clean feature discrepancy and local student variation through the same distance.
It does not establish that features are intrinsically superior to logits.

### How to interpret the logit-MSE result

The near tie between frozen-head logit MSE and feature anchoring should be reported rather than
minimized. It shows that most of the improvement comes from direct metric regression instead of
temperature tuning or probability-space matching. Accordingly, the paper should not claim that
only feature targets can work. The appropriate interpretation is:

> Direct regression accounts for most of the gain, and CFA instantiates this principle at the
> feature layer, where the teacher's representation remains directly compatible with the inherited
> classifier.

Thus, logit MSE is a control supporting the claim that softmax calibration and temperature search
are unnecessary, whereas the feature anchor is the concrete method developed in the paper.

### Why the teacher-epoch analysis belongs in the paper

Once the clean representation becomes the entire supervisory target, the choice of teacher checkpoint
is no longer incidental. This yields a testable prediction: if CFA simply transfers teacher accuracy,
a more accurate natural teacher should produce a more accurate student. The teacher ladder contradicts
this prediction. Increasing the teacher's training length raises its clean accuracy while moving the
student in the opposite direction on clean accuracy and changing its robustness.

This counterintuitive result should be developed as an empirical analysis of what the anchor transfers.
The class-geometry measurements can support the restrained conclusion that student behavior tracks the
teacher's class geometry more closely than its scalar clean accuracy. They should not be used to claim
that geometry has been established as the unique causal mechanism. The practical conclusion is that
teacher training length controls the student's clean--robust operating point, and maximum teacher
accuracy is not necessarily the appropriate checkpoint-selection criterion.

The exact correlation $r=-0.999$ should not carry the claim by itself because it is computed from a
small teacher ladder. The paper should emphasize the trend, matched student training, geometry
measurements, and replication across datasets or seeds when available.

### Why sensitivity-matched epsilon is part of the same story

Sensitivity-matched $\epsilon$ should not be introduced as an additional optimization trick. It follows
from making feature distance the sole training objective. The perturbation budget is specified in pixel
space, whereas the loss measures displacement in feature space; consequently, a uniform pixel radius
can induce very different changes in the anchor across samples. To first order,

$$
\Delta\mathcal L_i \approx \epsilon_i\lVert\nabla_x\mathcal L_i\rVert_1.
$$

This motivates allocating the fixed batch budget according to the input sensitivity of the anchor.
The conceptual statement is:

> Once representation distance becomes the training objective, the perturbation budget should be
> allocated according to the geometry induced by that objective.

The teacher-epoch analysis and sensitivity-matched radius therefore address two complementary questions:

* teacher training length determines **what geometry the anchor supplies**;
* sensitivity-matched $\epsilon$ determines **how that anchor is enforced across samples**.

They are consequences of treating the clean representation as the sole target, not independent modules
added to distinguish CFA from prior work.

### Recommended paper structure

1. **Preliminaries.** Define the teacher, student, threat set, feature map, and classifier.
2. **Do Natural Teachers Need Softened Targets?** Contrast teacher-only temperature, conventional
   shared-temperature KD, direct logit regression, and feature anchoring. Analyze why the two
   temperature formulations behave differently. Avoid presenting the rows as an additive sequence
   of improvements.
3. **A Clean Representation as the Sole Target.** Introduce the fixed clean-feature anchor and contrast
   its role with ARREST: sole objective rather than a regularizer, feature-based rather than CE-based
   attack generation, no label term, and an inherited fixed classifier.
4. **What the Anchor Controls.** Present $L\le F+O\le3L$ as a characterization of the fixed-target
   objective. Connect feature fidelity to the inherited classifier without claiming a classification
   certificate or feature superiority.
5. **What Does the Natural Teacher Transfer?** Present the teacher-epoch ladder and class-geometry
   measurements as an empirical analysis. Conclude that teacher accuracy alone does not predict the
   resulting student and that teacher training controls the operating point.
6. **Method: Clean Feature Anchoring.** State the full objective and implementation selected by the
   analysis.
7. **Method: Sensitivity-Matched Epsilon.** Derive the radius allocation from the mismatch between
   pixel-space budgets and feature-space loss sensitivity.
8. **Experiments and ablations.** Place the label-loss sweep, shared CE-attack control, alternative
   target comparisons, and other implementation controls here rather than making all of them equal
   steps in the Analysis narrative.

### Narrative discipline

The paper should not read as if CFA were discovered by eliminating every available alternative.
The design principle must precede the ablations:

> A non-robust teacher supplies a fixed reference at the clean input, while robustness is learned by
> constraining the student over the perturbation set.

Each main-text experiment should test one prediction of this principle. Temperature experiments test
whether a calibrated probability target is necessary; the adversarial-read control tests why the
teacher target must remain fixed; the label-loss sweep tests whether additional supervised loss is
needed; the teacher ladder tests what property of the natural teacher is transferred. Detailed variants
and exhaustive sweeps should remain in the experimental ablations or appendix.

The resulting high-level positioning is:

> CFA elevates the natural representation from an auxiliary regularizer to the sole adversarial
> training target, then adapts both teacher selection and perturbation allocation to the geometry of
> that target.

## 합의된 구조 (2026-09-09)

The revised claim above is adopted. Two things are added to it, and the page budget is deliberately
ignored for now: everything needed goes in, and the trimming happens afterwards.

### 1. The ladder keeps its spine

The four rows should not be sold as an additive sequence of improvements, but they are not four
arbitrary alternatives either: the middle two are connected by a limit. With the same temperature on
both sides the KD gradient decays as $1/\tau^2$, the conventional $\tau^2$ factor gives a finite
limit, and that limit is mean-removed logit regression -- measured at 0.00439 for both at
$\tau \ge 32$, and visible in the sweep as a plateau at $\tau = 16$ to $32$ (NRR 36.00, 36.11) just
below the logit MSE cell (36.43). With the teacher softened alone the gradient is $O(1)$ in $\tau$
and the target converges to the uniform distribution instead (maximum probability 0.0122 at
$\tau = 64$ against $1/C = 0.01$), which is why that curve turns over at $\tau = 16$ while the
conventional one does not.

So the section can say what the progression *is* -- raising the temperature on both sides is
linearising the softmax into a regression, and the last step removes it outright -- without claiming
each row is a contribution. That statement is what makes the ordering explanatory rather than a
ranking.

### 2. The attack objective is one of the four axes, not an implementation detail

Treating the clean representation as the sole target predicts that the same distance should generate
the adversarial examples, and ARREST is the natural contrast: it matches a representation but
generates its perturbations from cross-entropy alone. The 4x2 control measures what that choice is
worth -- the anchor loses 0.61 AutoAttack when the attack is taken away from it, the logit MSE 0.91,
conventional KD 0.74, teacher-only KD 0.25 -- and the ordering of objectives is unchanged under both
attacks, so the axis is separable from the target axis. This belongs in the section that contrasts
roles with ARREST (label term, attack objective, inherited classifier), not in the ablation dump.

### 3. Queue consequence

The factorial's stack half is dropped. The ladder does not need to be repeated under weight averaging
and AWP to make its point, and the cells that do matter for the main table are kept: conventional KD
and frozen-head logit MSE at the shipped recipe, which replace the teacher-only KD row currently
reported there. Roughly twenty hours of GPU return to the teacher-quality ladder, the table-5
remainder, and the epoch and learning-rate sensitivity cells.
