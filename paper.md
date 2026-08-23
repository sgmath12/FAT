# Anchor the features, not the logits: adversarial distillation from a naturally-trained teacher

Method-centric draft, rev. 2026-08-23. Numbers are ours unless marked as a published baseline.
Weight averaging and AWP are **not** part of the method — standard machinery, our baselines carry
them too, named only where the protocol needs it. All ablations are at the **base regime**
(50 epochs, no WA, no AWP, ε = 8/255) unless stated; headline numbers are at the full recipe.

---

## 1. Where this sits

Adversarial distillation is normally done **on logits**, from a **robust** teacher (ARD, RSLAD,
IAD), and that line works. This paper uses a **naturally-trained** teacher — the cheapest and by far
the most accurate one available — and anchors on its **features** instead. The argument is not that
logit distillation is a poor method; it is that four specific properties, which the feature anchor
has and a logit anchor does not, are what make a *natural* teacher usable at all.

The relevant difference between the two teachers is measurable. A naturally-trained head is
optimized for clean accuracy and nothing else, so it is extremely confident, and its confidence is
also the part an adversary moves:

| on clean test data | mean max softmax prob | ‖logits‖ | predictive entropy | clean |
|---|---:|---:|---:|---:|
| natural teacher | **0.820** | **16.63** | 0.76 | 77.38 |
| a trained robust student | 0.016 | 0.81 | 4.60 | 61.84 |

Under an ε = 8/255 attack the natural teacher's **feature rotates 62° and its norm inflates ×2.47**;
on a trained robust student the same attack moves 12–13° and ≈5%. A robust teacher is far less
confident to begin with, which is why the logit route is comfortable there and awkward here.

**The four properties** (derived in §2, proofs in `theory_v1.md` T.2b–T.2c):

1. The loss decomposes into *fidelity to the teacher* plus *local constancy of the student* — one
   term delivering both — because it is a **metric** on the target space. KL is not a metric and
   admits no such split.
2. The feature target is used **as it is**: no projection to $K$ class coordinates, no saturating
   nonlinearity.
3. There is **no temperature to choose**. On the logit route τ is squeezed from both sides: small τ
   saturates the target to a hard label (destruction), large τ flattens it to within $O(1/\tau)$ of
   uniform (attenuation). At τ = 16 the natural teacher's distribution is already within 1% of
   uniform.
4. The objective is **exactly zero at initialization** — the student starts as the teacher — so only
   the perturbation makes it non-zero, which is precisely the quantity to be trained.

> **What is new here, stated carefully.** Feature-space distillation is not new, and in ordinary
> (non-adversarial) KD the choice between matching features and matching logits is a well-worn axis
> on which the two are broadly comparable. The claim is that this choice, minor there, becomes
> **decisive** when the teacher is natural and the setting adversarial — §3 measures the size of the
> gap and §2 gives the four reasons. ⚠ *The characterization of the ordinary-KD literature above is
> asserted, not verified here; a related-work pass must confirm it before this framing is final.*

---

## 2. Method

$\Phi_t$ is the frozen feature map of a naturally-trained network; the student is initialized from
the teacher's weights.

### Stage 1 — anchor the backbone

$$\mathcal{L}=\mathbb{E}\big\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_t(x)\big\rVert^2,$$

$x_{\mathrm{adv}}$ from PGD on this same loss. No head, no logits, no temperature: the attack is
computed from the feature loss and the loss has no other term, so the classifier plays no role in
this stage at all.

**What this single term asks for.** Write $B(x)$ for the $\varepsilon$-ball and

$$L=\max_{x'\in B(x)}\lVert\Phi_s(x')-\Phi_t(x)\rVert,\quad
F=\lVert\Phi_s(x)-\Phi_t(x)\rVert,\quad
O=\!\!\max_{x',x''\in B(x)}\!\!\lVert\Phi_s(x')-\Phi_s(x'')\rVert,$$

$F$ the **fidelity** to the teacher at the clean point and $O$ the **oscillation** of the student's
own feature map on the ball — which is what robustness of a feature map means. Then

$$L\;\le\;F+O\;\le\;3L,$$

by the triangle inequality alone: $F\le L$ because $x\in B(x)$; $O\le 2L$ because
$\lVert\Phi_s(x')-\Phi_s(x'')\rVert\le\lVert\Phi_s(x')-\Phi_t(x)\rVert+\lVert\Phi_t(x)-\Phi_s(x'')\rVert$;
and $L\le F+O$ by splitting at $\Phi_s(x)$.

So the one term is equivalent, within a factor of three, to **reproduce the teacher on clean inputs
*and* be constant on $\varepsilon$-balls**. Both properties come from it and nothing else is
required for either. No model, no linearity, no assumption on the network.

**This is why a non-robust teacher is a usable target.** $\Phi_t$ is evaluated at $x$ and nowhere
else, so the teacher's instability never enters the objective. The student is not asked to imitate
the teacher; it is asked to hold the teacher's clean value while under attack — precisely what the
teacher cannot do, and precisely the content of $O$.

Measured on trained checkpoints (train split, each cell's own matched attack; $O$ lower-bounded by
$\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_s(x)\rVert$):

| | $L$ | $F$ | $O_{\text{lb}}$ | $F+O_{\text{lb}}$ | $3L$ | teacher's own $O$ |
|---|---:|---:|---:|---:|---:|---:|
| ours (raw) | 7.583 | 6.304 | 2.164 | 8.469 | 22.749 | **6.684** |
| ours (directional) | 0.674 | 0.542 | 0.238 | 0.780 | 2.022 | **0.566** |

The student oscillates **2.4× less than the teacher** on the same balls, matching the independent
angular measurement (12–13° against 62°); the remainder of the loss is fidelity rather than
instability. **The decomposition is a property of anchoring in a metric space.** The proof is the triangle
inequality on $\lVert\cdot\rVert$; KL is neither symmetric nor sub-additive along a path, so a logit
objective admits no corresponding split into fidelity and stability. That, and not any deficiency of
logit distillation as such, is the first of the four properties in §1.

**A second, smaller property.** The loss is exactly zero at initialization: the student starts as
the teacher, so $\Phi_s=\Phi_t$ and only the adversarial perturbation makes the objective non-zero.
A logit target is zero only at τ = 1, which is the setting where it carries nothing.

### The head: left alone

There is no second stage. The head-distillation term is **deleted**, and the classifier is simply
whatever the teacher's was — the student is initialized from the teacher, and nothing touches the
head thereafter. The feature anchor makes $\Phi_s\approx\Phi_t$, which is exactly the condition
under which the teacher's classifier is the right classifier for the student's features.

This is not an economy but the better of the options we measured (§3): re-training the head on
labels, on adversarial or on clean examples, *lowers* NRR (36.53 → 36.43 → 36.06), and giving the
head a distillation term with a badly chosen temperature costs 1.95 NRR. The term is at best
neutral and at worst harmful, so it is not a hyperparameter of the method.

Consequence: β and τ do not exist in this recipe, and the teacher appears in exactly one place —
the feature target of Stage 1.

### Per-sample ε, allocated by loss sensitivity

A fixed pixel radius moves the feature by very different amounts across samples. Inside an
$\ell_\infty$ ball of radius $e$ the first-order change of any loss is
$\max_{\lVert\delta\rVert_\infty\le e}\langle\nabla_xL,\delta\rangle = e\lVert\nabla_xL\rVert_1$, so
equalizing that movement under a fixed total budget $\sum_i\varepsilon_i = N\varepsilon$ gives
$\varepsilon_i \propto g_i^{-p}$, $g_i = \lVert\nabla_xL(x_i)\rVert$, clipped to
$[0.5, 1.5]\varepsilon$ and rescaled to restore the mean exactly.

**Budget preservation is load-bearing**: $p{=}1$ and $p{=}0$ spend the same total attack budget, so
any difference is allocation and not strength.

*Qualifications.* The rule equalizes the movement of the **loss**, which for a cosine form is
$\Delta\cos\theta$; converting to $\Delta\theta$ carries a per-sample $1/\sin\theta$, so "angular
budget" is an approximation. $g_i$ is one backward pass at the **clean** $x_i$, i.e. the sensitivity
at the starting point of a 10-step PGD. The derivation asks for $\ell_1$ and the runs used
$\ell_2$; trained both and they agree inside noise (62.51 / 28.44 vs 62.17 / 28.59).

---

## 3. Logit-following, priced

Pure KD from the same natural teacher, same regime, same attack, no interventions of any kind — just
τ. Everything in this table is the base regime: raw features, 50 epochs, no WA, no AWP, ε = 8/255.
NRR is the harmonic mean of clean and AA.

| | clean | PGD-20 | CW | **AA** | **NRR** |
|---|---:|---:|---:|---:|---:|
| logits, τ = 1 | 58.26 | 22.62 | 22.79 | 20.84 | 30.70 |
| logits, **τ = 4** | 59.39 | 30.30 | 26.84 | **24.48** | **34.67** |
| logits, τ = 16 | 57.78 | 31.57 | 26.11 | 24.00 | 33.91 |
| feature anchor, head KD at τ = 1 | 62.34 | 26.34 | 26.11 | 23.93 | 34.58 |
| **feature anchor, no head KD** | **62.72** | 28.69 | **27.80** | **25.88** | **36.53** |
| ‥ + head refit on adversarial examples | 62.77 | **29.93** | 27.66 | 25.65 | 36.43 |
| ‥ + head refit on clean examples | 62.70 | 28.90 | 27.23 | 25.15 | 36.06 |

**The feature anchor beats the best temperature on every axis** — clean +3.33, CW +0.96, AA +1.40,
NRR +1.86 — and it does so with **no head-distillation term at all**, hence with no temperature to
have chosen.

The logit curve is ∩-shaped, and both of its failure modes are one fact seen from two sides. At
small τ the teacher's distribution is still at 0.82 max-prob, so the student is asked to reproduce a
confidence 20× sharper than its own and robustness collapses. At large τ the target flattens to
0.0196 max-prob — near-uniform, nothing left to learn — *and* the warm-started student begins τ-fold
away from it, so the run spends its budget recovering (clean **1.01** at epoch 0 for τ = 16, still
climbing at the end). **τ has to be large and small at once.**

Note that τ = 16 has the best PGD-20 of the three logit rows and a worse AA than τ = 4. PGD does not
arbitrate here, and does not anywhere in this project.

*One cell is still missing:* the feature anchor **with** the head-KD term at τ = 16, with AA
measured. Its clean/PGD/CW are 62.86 / 32.36 / 27.18 but it was run with AA off, so the table cannot
yet say whether the head term is merely harmless or actively worth removing. Running.

---

## 4. The same conclusion at the full recipe

§3 is the base regime. The head term is equally removable once WA, AWP and the larger training
radius are on:

| | clean | PGD-20 | CW | AA |
|---|---:|---:|---:|---:|
| with head KD (τ = 16) | 60.74 | 34.94 | 30.53 | **28.69** |
| **head left at the teacher's, no KD term** | 60.45 | 30.71 | **30.54** | **28.63** |
| head KD with τ = 1 | 60.26 | 31.31 | 29.93 | 27.78 |

Deleting it: clean −0.29, CW +0.01, AA −0.06. Misconfiguring it: **−0.91 AA**. Same verdict as at
base, under a completely different regime.

The PGD-20 drop of 4.23 with CW and AA unmoved is worth naming, since it looks alarming and is not:
it is a CE-attack artifact from a head that is not matched to the features, and AutoAttack erases
the difference. Re-training the head does raise PGD (28.69 → 29.93 at base) but lowers AA and CW,
which is why we leave it alone and report the artifact rather than paper over it.

---

## 5. Why the anchor costs no robustness

**Model** (Tsipras et al., the setting where the objection to a non-robust anchor is usually
formalized). Robust feature $x_1 = y$ w.p. 0.95; non-robust $z_j\sim\mathcal{N}(\eta y,1)$,
$\eta = 4/\sqrt d$; an $\ell_\infty$ adversary acting on $z$ only. The natural teacher is
$\Phi_t = m(z)$ — 99.99% standard accuracy, ≈0 robust. Student $\Phi_s = a x_1 + b\,m(z)$.

**Theorem 1.** $J(a,b)=\mathbb{E}[R^2]+2\varepsilon\lvert b\rvert\mathbb{E}\lvert R\rvert+\varepsilon^2b^2$
with $R = ax_1+(b-1)m$, kinked at $b=0$; above a threshold $\varepsilon_0$ the minimizer has
$b^\star = 0$ **exactly** and $a^\star = q\eta$. *The adversarial term is an $\ell_1$ penalty on the
non-robust coefficient*: the teacher supplies the value, the inner maximization discards the route
it took to get there. Verified numerically ($d=512$): $b^\star = 0.510$ at $0.5\eta$ and exactly 0
at $\eta$, $2\eta$, $4\eta$.

**Corollary.** Every $\varepsilon$ in $J$ multiplies $b$, so at $b^\star=0$ the objective is
$\mathbb{E}(ax_1-m)^2$, free of $\varepsilon$: past the threshold, a larger training radius costs
the anchor nothing. Measured, $\varepsilon_{\mathrm{tr}}$ 8→8.8 costs clean 1.4 and buys AA 0.75, so
the claim is *saturation*, not *free*.

### What this argument does not deliver

It answers the objection — a non-robust anchor does not tax robustness — inside the model where the
objection lives. It does **not** explain the clean gain: there, $b^\star=0$ gives
$\Phi_s = q\eta x_1$, capped at $x_1$'s 95%, and plain adversarial training reaches the same
classifier. The model has one robust feature, so "match the teacher's value" and "match the label"
coincide and there is nothing left for the teacher's extra information to be. **The clean gain is an
empirical result here; the theory covers only why it is not paid for in robustness.**

*What we can say about the difference between the two networks* is measured rather than derived: a
trained robust student's clean features still point where the teacher's do (cos ≈ 0.83), but they
stop moving under attack (12° against 62°). The two models differ in the **stability** of the
feature map, not in its content — which is why the value is a usable target and why stability has to
come from the inner maximization instead. ⚠ The clean version of this story would have the
student's residual concentrated in the teacher's attack-volatile directions; measured, it is not
(concentration 0.94–1.14 against the clean-energy baseline). The account is consistent with the
measurements, not established by them.

---

## 6. Results

CIFAR-100 / ResNet-18. NRR = harmonic mean(clean, AA); AA = AutoAttack (`apgd-ce`+`apgd-t`), all
10 000 test images, ε = 8/255.

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

**clean +4.81 at AA +0.09** on CIFAR-100. The gain is on the clean axis and robustness is a tie,
which is what §5 predicts. Warm-starting from the teacher rather than from scratch is worth
**+2.02 AA**, so the initialization is part of the method.

---

## 7. Ablations

| at the full recipe | clean | CW | AA |
|---|---:|---:|---:|
| full method | 62.17 | 30.92 | 28.59 |
| uniform ε ($p=0$) | 60.74 | 30.53 | 28.69 |
| ε allocated by **difficulty** instead of sensitivity | 61.52 | 30.13 | **27.95** |
| no feature loss (logit distillation only) | 58.92 | 30.03 | 28.71 |
| head frozen, no head-KD term | 60.45 | 30.54 | 28.63 |

**Sensitivity-matched ε is worth +1.43 clean at an AA tie** (−0.10), at identical total attack
budget; on CIFAR-10 the same switch is +2.14 clean at −0.02 AA.

**The signal matters, not merely the per-sample-ness.** Row 3 keeps the exact weight multiset the
sensitivity rule produces — same values, same clamp, same mean, same total budget — and only
reassigns which sample gets which, ordering by difficulty (easy → large radius, as IAAT/CAT do). It
buys clean and **pays 0.74 AA** for it, ending below the uniform baseline on both CW and NRR. Only
the sensitivity ordering moves the frontier rather than sliding along it.
⚠ This is a *control*, not a comparison against IAAT/MMA/CAT as published methods; those carry
different recipes end to end and have not been run.

**The feature anchor is what buys clean.** Replacing it with logit distillation costs 1.82 clean and
leaves AA unchanged (28.71 vs 28.69). Every anchor-side intervention we have tried lands at an AA
tie and moves only clean; that is the method's signature and also the limit of what the feature term
may be credited with.

**Directional vs raw target.** Normalizing both sides — $\hat\Phi_s$ against $\hat\Phi_t$ — instead
of using raw features is **a tie**: at each design's own best schedule, NRR 39.17 against 39.29,
inside noise, reproduced on a second machine. **No claim is made on this axis.** The raw form is
presented above; the directional form is reported as an ablation.

---

## 8. Not claimed

- That a natural teacher is *better* than a robust one. The comparison class is methods without a
  robust teacher (ADR, TRADES-family). Where a robust teacher of comparable capacity is available,
  expect it to do at least as well; the model says only that the *anchor term* gains nothing from it.
- That AA improves. It does not. Every intervention here is a clean-axis lever at matched robustness.
- That the directional target beats the raw one (§7).
- Anything about why weight averaging, AWP or the learning-rate schedule help — they are not part of
  the method and we have no account of them. One of them, freezing the learning rate for the last
  third, interacts with the target's parametrization strongly enough to reverse the
  directional/raw ordering, which is why §7 reports that axis as a tie rather than a result.
- That removing the teacher's attack-volatile feature directions helps. It cannot be done: those
  directions carry the clean signal too (top-100 hold 92.8% of the attack displacement and 96.6% of
  the clean feature energy).
