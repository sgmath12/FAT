# Keep the teacher: feature anchoring recovers clean accuracy at no robustness cost

Draft, 2026-08-30. Written from `RESULTS.md`, `METHOD.md` §8–§9 and `theory_v1.md`. Every number is
ours unless marked as a published baseline. Weight averaging and AWP are standard machinery, carried
by our baselines too, and are not claimed here.

---

## 1. The observation

Adversarial training gives up clean accuracy. A naturally-trained network of the same architecture
has it — 77.65% on CIFAR-100 against an adversarially-trained 57.37% — and it is free to obtain. The
obvious move is to distil from it, and the obvious objection is equally old: a natural network's
accuracy rests on features an $\ell_\infty$ adversary can flip, so copying it should import that
fragility. This is why ADR self-anchors and why the ARD/RSLAD line requires a *robust* teacher.

The objection is measurable, and on the teacher it is exactly right. Under $\varepsilon=8/255$ the
natural teacher's feature **rotates 63.8° and its norm inflates ×2.45**; its robust accuracy is
$\approx0$. On our trained student the same attack moves 14.6°.

**What we find is that the fragility is not what gets inherited.** A student anchored to that
teacher's *features* keeps 83% of its representation (cosine similarity 0.8345) and pays nothing on
the robust axis. The clean accuracy comes along; the volatility does not.

---

## 2. The thesis, and a ladder that states it

> **Clean accuracy is how much of the teacher's representation the student still uses.** The
> method's content is that almost all of it can be kept without paying robustness.

Four ways of using the same clean teacher, same 100-epoch recipe, same initialization, differing in
one term. All start from the identical checkpoint, so they share a coordinate basis and
$\cos(\hat\Phi_s(x),\hat\Phi_t(x))$ is meaningful across rows. CIFAR-100, full test set.

| how the clean teacher is used | clean | PGD-10 | AA | $\cos$ vs teacher |
|---|---:|---:|---:|---:|
| not at all (ADR self-anchors) | 57.37 | 35.18 | 28.50 | *(different net)* |
| as an **initialization only**, then label-CE AT | 58.38 | 28.28 | — | 0.4703 |
| through its **logits** | 58.94 | 35.45 | 28.71 | **0.0968** |
| through its **features** | 60.76 | 35.17 | 28.69 | **0.8245** |
| + sensitivity-matched $\varepsilon$ | **62.16** | 35.06 | 28.59 | **0.8345** |
| the teacher itself | 77.65 | 0.00 | ≈0 | 1.0000 |

Two things to read off.

**Retention is a monotone axis and robustness is constant along all of it.** Among the four cells at
the shipped regime, clean accuracy and $\cos$ move together while PGD stays within 0.39 and AA
within $\pm0.11$ of 28.60.

**Neither an initialization nor a logit target keeps a student in the teacher.** Rows two and three
both *begin* at the teacher checkpoint and walk away: label-CE adversarial training ends at
$\cos=0.470$, logit distillation at $\cos=0.097$. Both feature maps are post-ReLU and hence
non-negative, so $0.097$ means near-disjoint support — **the logit-distilled student ends up using
different coordinates from the teacher it was distilled from.** Only the feature anchor stays
(0.82–0.83), and between the two regime-matched cells that is worth $+1.8$ clean at $-0.02$ AA.

*⚠ The AT clean-init row is a $\cos$ control only; that checkpoint is from an earlier 3-step / no-WA
regime and sits at PGD 28.28 against $\approx35$ elsewhere.*

Two questions follow, and only the second needs theory. *Why does keeping the teacher buy clean?* —
that is what the ladder measures. *Why is keeping a **non-robust** teacher free on robustness?* —
that is the surprising half, and §4 answers it.

---

## 3. Method

$\Phi_t$ is the frozen feature map of a naturally-trained network; the student is initialized from
its weights.

$$\mathcal{L}=\mathbb{E}\big\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_t(x)\big\rVert^2,$$

with $x_{\mathrm{adv}}$ from PGD on this same loss. **That is the entire objective.** No head
distillation, no temperature, no loss weight: the attack is computed from the feature loss, the loss
has no other term, and the classifier is left at the teacher's throughout. §7 shows each of those
deletions is measured, not assumed.

**Per-sample $\varepsilon$, allocated by loss sensitivity.** A fixed pixel radius moves the feature
by very different amounts across samples. Inside an $\ell_\infty$ ball of radius $e$ the first-order
change of any loss is $\max_{\lVert\delta\rVert_\infty\le e}\langle\nabla_xL,\delta\rangle
= e\lVert\nabla_xL\rVert_1$, so equalizing that movement under a fixed total budget
$\sum_i\varepsilon_i = N\varepsilon$ gives $\varepsilon_i \propto g_i^{-p}$ with
$g_i=\lVert\nabla_xL(x_i)\rVert$, clipped to $[0.5,1.5]\varepsilon$ and rescaled to restore the mean
exactly. **Budget preservation is load-bearing**: $p{=}1$ and $p{=}0$ spend the same total attack
budget, so any difference between them is allocation and not strength.

*Qualifications.* The rule equalizes the movement of the **loss**, not an angle; $g_i$ is one
backward pass at the clean $x_i$, so it is the sensitivity at the starting point of a 10-step PGD.
The derivation asks for $\ell_1$ and our runs used $\ell_2$; we trained both and they agree inside
noise (62.51 / AA 28.44 against 62.17 / 28.59).

---

## 4. Why anchoring to a non-robust teacher is free

### 4.1 The objective asks for fidelity *and* stability, and the teacher is never evaluated under attack

Write $B(x)$ for the $\varepsilon$-ball and

$$L=\max_{x'\in B(x)}\lVert\Phi_s(x')-\Phi_t(x)\rVert,\quad
F=\lVert\Phi_s(x)-\Phi_t(x)\rVert,\quad
O=\!\!\max_{x',x''\in B(x)}\!\!\lVert\Phi_s(x')-\Phi_s(x'')\rVert.$$

**Proposition 1.** $L\le F+O\le 3L$.

*Proof.* $F\le L$ since $x\in B(x)$. For $x',x''\in B(x)$,
$\lVert\Phi_s(x')-\Phi_s(x'')\rVert\le\lVert\Phi_s(x')-\Phi_t(x)\rVert+\lVert\Phi_t(x)-\Phi_s(x'')\rVert\le2L$,
so $O\le2L$. Conversely $\lVert\Phi_s(x')-\Phi_t(x)\rVert\le O+F$ for every $x'$. $\square$

The triangle inequality and nothing else: no model, no linearity, no assumption on the network. One
term is equivalent within a factor of three to **reproduce the teacher on clean inputs** *and* **be
constant on $\varepsilon$-balls**, which is what robustness of a feature map means.

**This is why a non-robust teacher is usable.** $\Phi_t$ appears at $x$ and nowhere else, so the
teacher's own instability never enters the objective. The student is not asked to imitate the
teacher under attack; it is asked to hold the teacher's clean value *while* under attack — precisely
what the teacher cannot do, and precisely the content of $O$.

Measured on trained checkpoints (train split, matched attack; $O$ lower-bounded by
$\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_s(x)\rVert$):

| | $L$ | $F$ | $O_{\text{lb}}$ | teacher's own $O$ |
|---|---:|---:|---:|---:|
| ours (raw target) | 7.583 | 6.304 | 2.164 | **6.684** |
| ours (normalized target) | 0.674 | 0.542 | 0.238 | **0.566** |

The student oscillates **2.4× less than the teacher** on the same balls, matching the independent
angular measurement (14.6° against 63.8°). Most of what remains in the loss is fidelity.

A second, smaller property: the objective is **exactly zero at initialization**, since the student
starts as the teacher. Only the perturbation makes it non-zero — which is the quantity one wants to
train.

### 4.2 The non-robust component is zeroed, not shrunk

In the Tsipras et al. model — robust feature $x_1=y$ w.p. $p$, non-robust $z_j\sim\mathcal N(\eta y,1)$,
an $\ell_\infty$ adversary acting on $z$ only, natural teacher $\Phi_t=m(z)$ — take
$\Phi_s = a x_1 + b\,m(z)$.

**Proposition 2.** With the inner maximum taken exactly,
$J(a,b)=\mathbb{E}[R^2]+2\varepsilon\lvert b\rvert\,\mathbb{E}\lvert R\rvert+\varepsilon^2b^2$
where $R=ax_1+(b-1)m$. $J$ is kinked at $b=0$ with subgradient jump
$2\varepsilon\mathbb{E}\lvert R(a,0)\rvert>0$, so above a threshold $\varepsilon_0$ the minimizer has
$b^\star=0$ **exactly** and $a^\star=q\eta$.

**The adversarial term is an $\ell_1$ penalty on the non-robust coefficient.** The teacher supplies
the value to be matched; the inner maximization discards the route the teacher took to it. Verified
numerically ($d=512$): $b^\star=0.510$ at $\varepsilon=0.5\eta$ and exactly 0 at $\eta,2\eta,4\eta$,
so $\varepsilon_0\in(0.5\eta,\eta)$ and the operating radius is far above it.

**Corollary.** Every $\varepsilon$ in $J$ multiplies $b$, so at $b^\star=0$ the objective is
$\mathbb{E}(ax_1-m)^2$, free of $\varepsilon$: past the threshold, a larger training radius costs the
anchor nothing. Measured, $\varepsilon_{\mathrm{tr}}$ $8\to8.8$ costs 1.4 clean and buys 0.75 AA on
CIFAR-100, and on Tiny-ImageNet raising it further leaves the clean penalty **flat at $-2.4$ for the
whole run** rather than widening. The claim is *saturation*, not *free*.

### 4.3 What the loss must be small relative to

Proposition 1 controls $F$ and $O$, but $O$ alone is **not** a sufficient statistic for robust
accuracy: across the teacher ladder (§6) it correlates with AA at $r=+0.83$ — the wrong sign.
Teachers that oscillate *more* produce students that are *more* robust. The pricing that fixes this
follows from the head being frozen at the teacher's.

**Proposition 3.** Let $\gamma_t(x)=\min_{c\ne y}\langle w_y-w_c,\Phi_t(x)\rangle$ be the teacher's
clean margin and $D_y=\max_{c\ne y}\lVert w_y-w_c\rVert$. If $\gamma_t(x)>D_y L(x)$ then $x$ is
robustly correct.

*Proof.* For any $x'\in B(x)$ and $c\ne y$,
$\langle w_y-w_c,\Phi_s(x')\rangle=\langle w_y-w_c,\Phi_t(x)\rangle+\langle w_y-w_c,\Phi_s(x')-\Phi_t(x)\rangle
\ge\gamma_t(x)-D_yL(x)$ by Cauchy–Schwarz. $\square$

Across the teacher ladder the quantity this names tracks **both** axes at once:

$$r\big(D L/\gamma_t,\ \mathrm{AA}\big)=-0.971,\qquad r\big(D L/\gamma_t,\ \text{clean}\big)=+0.977.$$

$\mathbb{E}[\gamma_t]$ is flat across the ladder (4.20→4.57), so the ratio moves because $L$ falls:
**longer-trained teachers are easier to match under attack.**

⚠ The certificate is **vacuous** — it certifies 1.5–7.9% against a measured AA of 24–26, because
Cauchy–Schwarz assumes an alignment between the displacement and $w_y-w_c$ that does not occur
($\mathbb{E}[L]\approx7.8$ against $\mathbb{E}[\gamma_t]\approx4.5$). It is reported for the quantity
it identifies, not as a bound that binds, and it does not reproduce the saturation of AA after 150ep.

---

## 5. Why plain adversarial training does not do this

If keeping the teacher's geometry is free, why does AT give it up? Because it is never asked to keep
it, and the cheapest route to stability is to stop reading the volatile directions.

**Setup.** $x\mid y\sim\mathcal N(\eta y,I)$ on $\mathbb{R}^D$; coordinate $j$ has reliability
$\eta_j$, and $\eta_j/\varepsilon$ is all that matters. For a linear readout,
$M_c=c^\top\eta/\lVert c\rVert_2$ and $M_r=(c^\top\eta-\varepsilon\lVert c\rVert_1)/\lVert c\rVert_2$.

**Proposition 4 (AT is a hard threshold).** $\arg\max_c M_r\propto(\eta-\varepsilon)_+$.
*Proof.* Cauchy–Schwarz, with equality iff $c\propto(\eta-\varepsilon)_+$. $\square$
Not an artifact of optimizing accuracy rather than a surrogate: the adversarial **logistic** loss
returns the same solution (support 64 vs 64, correlation 1.0000).

**Every coordinate with $\eta_j\le\varepsilon$ gets weight exactly zero, however close $\eta_j$ is to
$\varepsilon$. Usefulness is continuous; the robust treatment of it is not.**

**Proposition 5 (what the discontinuity costs).** Restoring weight to a deleted coordinate trades at
$\frac{dM_c}{-dM_r}=\frac{\eta_j}{\varepsilon-\eta_j}\to\infty$ as $\eta_j\uparrow\varepsilon$.
Verified against finite differences (67.28/67.22 at $\eta_j/\varepsilon=0.986$; 4.01/4.01 at 0.812).
Relaxing the threshold traces a frontier on which clean rises several points for a robustness cost
inside any realistic noise band.

**Why the existing model could not see this.** Proposition 2's spectrum has two atoms and **no mass
near $\eta_j/\varepsilon=1$**, where the entire effect lives; relaxing its threshold changes nothing
until the whole bulk returns at once and robustness collapses 22 points. **That model rules out the
paper's effect by construction** — which is exactly why it finds anchoring and plain AT to be the
same classifier, and why §4.2 explains only the *free* half.

⚠ **The limit of every re-weighting account.** For a linear readout,
$M_c(c)-M_r(c)=\varepsilon\lVert c\rVert_1/\lVert c\rVert_2$ *exactly*: reading a coordinate and
being exposed on it are the same parameter, so deletion is the only gap-reducing move a linear map
has, and across four spectra × six radii the anchored objective's own optimum trades at 0.46–2.6
against a measured rate $\ge12$. **A nonlinear map escapes this** because its exposure is $O$, which
is not a function of how many coordinates it reads: it can depend on an attackable coordinate and
still be locally constant in it. That is what §4.1's $F+O$ asks for and adversarial CE does not.

---

## 6. A dial that comes free: teacher training length

The teacher is naturally trained, so its geometry can be moved at no adversarial cost. Changing only
the checkpoint path — student configuration untouched:

| teacher | T.clean | T. $S_w/S_b$ | **S.clean** | **S.AA** | NRR |
|---|---:|---:|---:|---:|---:|
| 50ep | 75.81 | 1.100 | **64.28** | 24.38 | 35.35 |
| 100ep | 76.62 | 0.982 | 63.65 | 25.20 | 36.11 |
| 150ep | 77.52 | 0.887 | 62.93 | 25.78 | 36.51 |
| 200ep | 77.65 | 0.808 | 62.72 | **25.88** | **36.64** |
| 300ep | 78.32 | 0.712 | 62.24 | 25.80 | 36.48 |

$$r(\text{teacher clean},\ \text{student clean}) = \mathbf{-0.999}.$$

The teacher gains 2.51 points and the student **loses** 2.04. **A more accurate teacher makes a
worse student**; what the student inherits is not the teacher's accuracy. NRR peaks in the middle,
so there is an optimal teacher length rather than "longer is better".

Reproduced on Tiny-ImageNet, where it is *accuracy-controlled*: two teachers 0.32 apart (65.97 /
66.29) move the student by 1.92 clean and 1.58 AA. Accuracy cannot be the cause there.

**What actually transfers is class separation, not concentration.** Splitting $S_w/S_b$ on the unit
sphere:

| | teacher 50→300ep | student 50→300ep |
|---|---:|---:|
| sample → own class mean | −3.7° | **+0.3° (flat)** |
| class mean ↔ nearest class mean | **+9.2°** | **+4.4°** |

$r(\text{teacher gap},\text{student gap})=+0.999$. The student copies the separation and not the
concentration, so "the teacher collapses" is the wrong description.

**One change moves both axes in opposite directions.** Clean falls because the student realizes only
48% of the teacher's separation and the shortfall widens, so $\cos(\hat\Phi_s,\hat\Phi_t)$ falls
0.8607→0.8311 and clean tracks it at $r=+0.994$. AA rises because the margin it *does* gain (+4.4°)
outpaces the attack's rotation (+1.9°); rotation priced against margin correlates with AA at
$-0.991$ and saturates where AA saturates.

**The robust-teacher line does not have this knob** — there the teacher's geometry is fixed by
adversarial training. ⚠ This must be stated as *"a teacher-geometry axis opens at natural-training
cost"*, never as *"a robust teacher is unnecessary"*, which is a ranking claim we do not test.

---

## 7. Results

CIFAR-100 / ResNet-18. NRR is the harmonic mean of clean and AA; AA is AutoAttack (`apgd-ce`+`apgd-t`)
on all 10 000 test images at $\varepsilon=8/255$.

| | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **Ours** | **62.17** | 34.77 | **30.92** | **28.59** | **39.17** |
| ADR + WA + AWP | 57.36 | 34.92 | 30.62 | 28.50 | 38.08 |

CIFAR-10 / ResNet-18, **identical configuration** — the two differ only in dataset and teacher
checkpoint, with no per-dataset tuning:

| | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **Ours** | **84.66** | 56.74 | **53.94** | 51.87 | **64.33** |
| ADR + WA + AWP | 83.26 | — | — | 51.18 | 63.39 |
| CURE | 86.76 | 54.92 | 52.48 | 49.69 | 63.19 |

Tiny-ImageNet-200 / ResNet-18, same recipe transferred unchanged:

| | clean | AA | NRR |
|---|---:|---:|---:|
| **Ours** (200ep teacher) | **55.16** | **20.54** | **29.93** |
| Ours (80ep teacher) | 57.08 | 18.96 | 28.46 |
| ADR + WA + AWP | 48.27 | 20.10 | 28.38 |

**clean +4.81 at AA +0.09** on CIFAR-100, and on Tiny-ImageNet **+6.89 clean at +0.44 AA** — the
first cell in this project to win both axes outright. Warm-starting from the teacher rather than
from scratch is worth **+2.02 AA**, so the initialization is part of the method.

*Protocol.* Our runs and the ADR rows both use WA and AWP. ADR trains 80 epochs on Tiny-ImageNet and
we train 100; our Tiny-ImageNet teacher is itself only 80 epochs (clean 65.97) and had not converged.

---

## 8. Ablations

**Logit anchoring, priced.** Same teacher, same regime, same attack, no interventions — only $\tau$.
Base regime: raw features, 50 epochs, no WA, no AWP, $\varepsilon=8/255$.

| | clean | PGD-20 | CW | **AA** | **NRR** |
|---|---:|---:|---:|---:|---:|
| logits, $\tau=1$ | 58.26 | 22.62 | 22.79 | 20.84 | 30.70 |
| logits, **$\tau=4$** | 59.39 | 30.30 | 26.84 | **24.48** | **34.67** |
| logits, $\tau=16$ | 57.78 | 31.57 | 26.11 | 24.00 | 33.91 |
| feature anchor, head KD at $\tau=1$ | 62.34 | 26.34 | 26.11 | 23.93 | 34.58 |
| feature anchor, head KD at $\tau=16$ | 62.61 | **32.47** | 27.32 | 25.61 | 36.35 |
| **feature anchor, no head KD** | **62.72** | 28.69 | **27.80** | **25.88** | **36.64** |
| ‥ + head refit, adversarial CE | 62.77 | 29.93 | 27.66 | 25.65 | 36.42 |
| ‥ + head refit, adv CE + smoothing 0.1 | 62.57 | 30.43 | 27.55 | 25.66 | 36.39 |
| ‥ + head refit, clean CE | 62.70 | 28.90 | 27.23 | 25.15 | 35.90 |

**The feature anchor beats the best temperature on every axis** (+3.33 clean, +0.96 CW, +1.40 AA,
+1.97 NRR) and has no temperature to have chosen. The logit curve is $\cap$-shaped because $\tau$ is
squeezed from both sides: at $\tau=1$ the teacher is still at 0.820 max-prob against the student's
0.016, so the student is asked for a confidence 20× sharper than its own; at $\tau=16$ the target is
at 0.0196, within 1% of uniform (0.0100), *and* the warm-started student begins 16-fold away from it
— collapsing to clean 1.01 in the first epoch. **$\tau$ has to be large and small at once.**

**The head term is worth removing, not tuning**: no temperature beats deleting it (36.35 at the best
$\tau$, 36.64 with the term gone), and every form of head re-training lands below leaving the
teacher's head untouched. Label smoothing lands exactly where hard labels do, so "hard labels are too
sharp for this stage" is not the explanation. The same verdict holds at the full recipe (deleting the
head KD costs $-0.06$ AA; misconfiguring it costs $-0.91$).

**Sensitivity-matched $\varepsilon$, and the signal matters.** At the full recipe $p{=}1$ is worth
+1.43 clean at $-0.10$ AA on CIFAR-100 and +2.14 at $-0.02$ on CIFAR-10, at identical total attack
budget. Keeping the *exact* weight multiset the sensitivity rule produces and only reassigning which
sample gets which — ordering by difficulty, as IAAT/CAT do — buys clean and **pays 0.74 AA**, ending
below the uniform baseline on both CW and NRR. Only the sensitivity ordering moves the frontier
rather than sliding along it. ⚠ This is a *control*, not a comparison against IAAT/MMA/CAT as
published methods, which have not been run.

**PGD would have inverted most of this table.** Head KD at $\tau=16$ has the best PGD-20 (32.47) and a
worse AA than the row with no head term at all (25.61 vs 25.88); across the label-smoothing sweep PGD
rises monotonically while CW falls. Only AutoAttack arbitrates.

---

## 9. What is not claimed

- **That a natural teacher is better than a robust one.** The comparison class is methods without a
  robust teacher (ADR, TRADES-family). Where a robust teacher of comparable capacity is available,
  expect it to do at least as well; our results say the *anchor term* gains nothing from it.
- **That AA improves.** On CIFAR it does not — every intervention is a clean-axis lever at matched
  robustness. Tiny-ImageNet with a 200ep teacher is the one cell that gains on both.
- **That the directional and raw targets differ.** Normalizing both sides is a tie: at each design's
  own best schedule, NRR 39.17 against 39.29, inside noise, reproduced on a second machine.
- **That §5's model predicts the size of the effect.** It predicts the *shape* — a discontinuous
  threshold and a diverging exchange rate near it. Across the anchored cells a $\cos$ change of 0.017
  accompanies +2.4 clean while a 0.73 collapse accompanies only $-1.8$, so the relation is an
  ordering, not a dose–response.
- **Anything about why WA, AWP or the learning-rate schedule help.** They are not part of the method
  and we have no account of them. One of them — freezing the learning rate for the last third —
  interacts with the target's parametrization strongly enough to reverse the directional/raw ordering,
  which is why that axis is reported as a tie.
- **That removing the teacher's attack-volatile directions helps.** It cannot be done: the top-100
  volatile directions carry 92.8% of the attack displacement and 96.6% of the clean feature energy.

## 10. Open

**Related work.** The claim that feature-vs-logit is a tuning decision in ordinary KD and a
structural one here rests on our gap being far larger than the usual one. Establishing the usual one
is a prerequisite, not a nicety.

**The saturation.** AA flattens after a 150ep teacher while $DL/\gamma_t$ keeps falling; the
hand-picked rotation-over-class-gap ratio flattens with it. Whether that is Cauchy–Schwarz slack or a
sign that the right denominator is the class-mean gap is unresolved.

**The Tiny-ImageNet ladder has not peaked.** NRR is still rising at the longest teacher we have
trained (28.46 → 28.57 → 29.93), so the optimum there is above 200 epochs and unmeasured.
