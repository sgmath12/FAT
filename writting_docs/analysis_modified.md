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
