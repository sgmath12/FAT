# Anchor the features of a naturally-trained network

Draft, 2026-08-30, rev. 2. Sources: `RESULTS.md`, `METHOD.md` §8–§9, `theory_v1.md`,
`comparison_resnet18_cifar100.md`. Numbers are ours unless marked as published. Weight averaging and
AWP are standard machinery carried by our baselines too and are not claimed here.

---

## 1. Result

Adversarial training trades clean accuracy for robustness, and on CIFAR-100 / ResNet-18 the
published frontier has been stuck in a narrow band: methods that reach AA $\approx28.5$ sit at clean
$\approx57$, and methods that reach clean $\approx63$ sit at AA $\approx24$.

| | clean | AA | NRR |
|---|---:|---:|---:|
| B-MTARD (robust WRN-70-16 teacher) | **65.08** | 23.98 | 35.05 |
| Generalist++ | 62.97 | 23.96 | 34.71 |
| **Ours** | **62.17** | **28.59** | **39.17** |
| ADR + WA + AWP | 57.36 | 28.50 | 38.08 |
| Consistency-AT + RPAT | 60.33 | 26.31 | 36.64 |
| PGD-AT | 56.56 | 25.02 | 34.69 |

**No published ResNet-18 / CIFAR-100 result dominates ours on both axes.** Read either way, the gap
is the same size: at matched robustness we are **+4.81 clean** over the strongest AA method, and at
matched clean accuracy we are **+4.63 AA** over the strongest clean methods. NRR 39.17 is +1.09 over
the previous best.

The method is one loss term. It uses a **naturally-trained** teacher — no adversarial training, no
robust teacher, no extra network — and has **no hyperparameter to select**: no temperature, no loss
weight, no head training. The same configuration transfers to CIFAR-10 and Tiny-ImageNet with only
the dataset name and teacher checkpoint changed.

Two things make it work, and the second was not expected.

**(i) The teacher is never evaluated under attack.** The objection to distilling from a natural
network is that its accuracy rests on fragile features. Our loss matches the teacher's *clean*
feature at the *adversarial* input, so the teacher's own instability never enters the objective
(§3). The student ends up 2.4× more stable than the teacher on the same $\varepsilon$-balls.

**(ii) What transfers is the teacher's geometry, not its accuracy.** Across a ladder of teachers
trained for 50–300 epochs, $r(\text{teacher clean},\ \text{student clean}) = \mathbf{-0.999}$ — **a
more accurate teacher produces a worse student** (§4). Since the teacher is naturally trained, its
training length becomes a free trade-off dial, one the robust-teacher distillation line does not
have.

---

## 2. Method

$\Phi_t$ is the frozen feature map of a naturally-trained network; the student is initialized from
its weights and trained with

$$\mathcal{L}=\mathbb{E}\big\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_t(x)\big\rVert^2,\qquad
x_{\mathrm{adv}}=\arg\max_{x'\in B(x,\varepsilon)}\lVert\Phi_s(x')-\Phi_t(x)\rVert .$$

That is the whole objective. The classifier stays at the teacher's and is never trained; there is no
logit distillation term, hence no temperature and no loss weight. §7 shows each of those deletions is
measured rather than assumed — every form of head training we tried scores *below* leaving it alone.

**Per-sample $\varepsilon$, allocated by loss sensitivity.** A fixed pixel radius moves the feature
by very different amounts across samples. Inside an $\ell_\infty$ ball of radius $e$ the first-order
change of any loss is $\max_{\lVert\delta\rVert_\infty\le e}\langle\nabla_xL,\delta\rangle=e\lVert\nabla_xL\rVert_1$,
so equalizing that movement subject to a fixed total budget $\sum_i\varepsilon_i=N\varepsilon$ gives
$\varepsilon_i\propto g_i^{-p}$, $g_i=\lVert\nabla_xL(x_i)\rVert$, clipped to $[0.5,1.5]\varepsilon$
and rescaled to restore the mean exactly. **Budget preservation is what makes the comparison mean
anything**: $p{=}1$ and $p{=}0$ spend the same total attack budget, so any difference is allocation,
not strength.

*Qualifications.* The rule equalizes the movement of the **loss**, not of an angle. $g_i$ is a single
backward pass at the clean $x_i$, i.e. the sensitivity at the starting point of a 10-step PGD. The
derivation asks for $\ell_1$ and our runs used $\ell_2$; both were trained and agree inside noise
(62.51 / AA 28.44 against 62.17 / 28.59).

---

## 3. Why anchoring to a non-robust teacher costs no robustness

One result carries this section. It needs no model and no assumption on the network, and it is the
only place the paper's central surprise is actually explained.

Write $B(x)$ for the $\varepsilon$-ball and

$$L=\max_{x'\in B(x)}\lVert\Phi_s(x')-\Phi_t(x)\rVert,\quad
F=\lVert\Phi_s(x)-\Phi_t(x)\rVert,\quad
O=\!\!\max_{x',x''\in B(x)}\!\!\lVert\Phi_s(x')-\Phi_s(x'')\rVert,$$

so $F$ is fidelity to the teacher at the clean point and $O$ is the oscillation of the student's own
map on the ball.

**Proposition 1.** $L\le F+O\le 3L$.

*Proof.* $F\le L$ since $x\in B(x)$; $O\le2L$ by inserting $\Phi_t(x)$; and
$\lVert\Phi_s(x')-\Phi_t(x)\rVert\le O+F$ by inserting $\Phi_s(x)$. $\square$

**The teacher appears at $x$ and nowhere else.** That single structural fact is the answer to the
standing objection. The student is never asked to reproduce anything the teacher does *off* the
clean point, so the teacher's own fragility — 63.8° of rotation and ×2.45 of norm inflation under
$\varepsilon=8/255$ — is not in the objective to be inherited. What the student is asked for is to
hold the teacher's clean value *while under attack*, which is exactly what the teacher cannot do, and
exactly the content of $O$.

Measured on trained checkpoints, the student ends up **2.4× more stable than its own teacher** on the
same balls:

| | $L$ | $F$ | $O_{\text{lb}}$ | teacher's own $O$ |
|---|---:|---:|---:|---:|
| ours (raw target) | 7.583 | 6.304 | 2.164 | **6.684** |
| ours (normalized target) | 0.674 | 0.542 | 0.238 | **0.566** |

matching the independent angular measurement (14.6° against 63.8°). Most of what remains in the loss
is fidelity, not instability.

A second, smaller consequence: the objective is **exactly zero at initialization**, since the student
starts as the teacher. Only the perturbation makes it non-zero — which is precisely the quantity to
be trained. A logit target is zero only at $\tau=1$, which is where it carries no information (§7).

⚠ **What Proposition 1 does not do.** It bounds $F+O$, and $O$ is *not* a sufficient statistic for
robust accuracy: across the teacher ladder of §4 it correlates with AA at $r=+0.83$ — the wrong sign,
with the more volatile teachers producing the more robust students. The proposition explains why the
anchor is *free*, and should not be read as explaining how much robustness the objective buys. Two
partial results address that and neither is conclusive; both are in Appendix A, and the honest
summary is that **the quantity robust accuracy actually tracks is $L$ measured against a margin, and
we can derive that quantity but not a tight bound on it.**

## 4. What transfers is geometry, not accuracy

The teacher is naturally trained, so its geometry can be moved at no adversarial cost. Changing only
the checkpoint path, with the student configuration untouched:

| teacher | T.clean | T. $S_w/S_b$ | **S.clean** | **S.AA** | NRR |
|---|---:|---:|---:|---:|---:|
| 50ep | 75.81 | 1.100 | **64.28** | 24.38 | 35.35 |
| 100ep | 76.62 | 0.982 | 63.65 | 25.20 | 36.11 |
| 150ep | 77.52 | 0.887 | 62.93 | 25.78 | 36.51 |
| 200ep | 77.65 | 0.808 | 62.72 | **25.88** | **36.64** |
| 300ep | 78.32 | 0.712 | 62.24 | 25.80 | 36.48 |

$$r(\text{teacher clean},\ \text{student clean})=\mathbf{-0.999}.$$

The teacher gains 2.51 points of clean accuracy across the ladder and the student **loses** 2.04.
NRR peaks in the middle, so there is an optimal teacher length rather than "longer is better", and
finding it costs five natural-training runs and no adversarial ones.

**Reproduced on Tiny-ImageNet, where it is accuracy-controlled.** Two teachers 0.32 apart (65.97 /
66.29) move the student by **1.92 clean and 1.58 AA**. Teacher accuracy cannot be the cause there.

**What moves is class separation, not within-class concentration.** Splitting $S_w/S_b$ on the unit
sphere:

| | teacher 50→300ep | student 50→300ep |
|---|---:|---:|
| sample → own class mean | −3.7° | **+0.3° (flat)** |
| class mean ↔ nearest class mean | **+9.2°** | **+4.4°** |

$r(\text{teacher gap},\ \text{student gap})=+0.999$: the student copies the separation and not the
concentration, so "the teacher collapses" is the wrong description of the mechanism.

**One change moves both axes in opposite directions.** Clean falls because the student realizes only
48% of the teacher's separation and the shortfall widens, so $\cos(\hat\Phi_s,\hat\Phi_t)$ falls
0.8607→0.8311 and clean tracks it at $r=+0.994$. AA rises because the margin the student *does* gain
(+4.4°) outpaces the attack's rotation (+1.9°); rotation priced against margin correlates with AA at
$-0.991$ and flattens exactly where AA flattens.

⚠ Stated as *"a teacher-geometry axis opens at natural-training cost"*, never as *"a robust teacher
is unnecessary"* — the latter is a ranking claim against a line we do not compete with or test.

---

## 5. Why the anchor is needed: retention, and why plain AT gives it up

### 5.1 An initialization does not hold, and a logit target does not retain

Four ways of using the same clean teacher, same 100-epoch recipe, same initialization, differing in
one term — so all rows share a coordinate basis and $\cos(\hat\Phi_s,\hat\Phi_t)$ is comparable.

| how the clean teacher is used | clean | PGD-10 | AA | $\cos$ vs teacher |
|---|---:|---:|---:|---:|
| not at all (ADR self-anchors) | 57.37 | 35.18 | 28.50 | *(different net)* |
| as an **initialization only**, then label-CE AT | 58.38 | 28.28 | — | 0.4703 |
| through its **logits** | 58.94 | 35.45 | 28.71 | **0.0968** |
| through its **features** | 60.76 | 35.17 | 28.69 | **0.8245** |
| + sensitivity-matched $\varepsilon$ | **62.16** | 35.06 | 28.59 | **0.8345** |
| the teacher itself | 77.65 | 0.00 | ≈0 | 1.0000 |

Rows two and three both *begin* at the teacher checkpoint and walk away from it. Both feature maps
are post-ReLU and hence non-negative, so $\cos=0.097$ means near-disjoint support: **the
logit-distilled student ends up using different coordinates from the teacher it was distilled from.**
Only the feature anchor stays, and between the two regime-matched cells that is worth $+1.8$ clean at
$-0.02$ AA. Meanwhile robustness is constant across all four: PGD within 0.39, AA within $\pm0.11$.

⚠ The AT clean-init row is a $\cos$ control only; that checkpoint is from an earlier regime and sits
at PGD 28.28 against $\approx35$ elsewhere. ⚠ The relation is an **ordering, not a dose–response**: a
$\cos$ change of 0.017 accompanies $+2.4$ clean while the 0.73 collapse accompanies only $-1.8$.

### 5.2 Robust optimality is a hard threshold, and deletion is the only move a linear map has

**Setup.** $x\mid y\sim\mathcal N(\eta y,I)$ on $\mathbb{R}^D$; coordinate $j$ has reliability
$\eta_j$ and $\eta_j/\varepsilon$ is all that matters. For a linear readout,
$M_c=c^\top\eta/\lVert c\rVert_2$ and $M_r=(c^\top\eta-\varepsilon\lVert c\rVert_1)/\lVert c\rVert_2$.

**Proposition 4.** $\arg\max_cM_r\propto(\eta-\varepsilon)_+$ (Cauchy–Schwarz, equality iff
proportional). The adversarial *logistic* loss returns the same solution (support 64 vs 64,
correlation 1.0000), so this is not an artifact of optimizing accuracy instead of a surrogate.

**Every coordinate with $\eta_j\le\varepsilon$ receives weight exactly zero, however close $\eta_j$ is
to $\varepsilon$: usefulness is continuous, the robust treatment of it is not.**

**Proposition 5.** Restoring weight to a deleted coordinate trades at
$dM_c/(-dM_r)=\eta_j/(\varepsilon-\eta_j)\to\infty$ as $\eta_j\uparrow\varepsilon$. Verified against
finite differences (67.28 / 67.22 at $\eta_j/\varepsilon=0.986$; 4.01 / 4.01 at 0.812).

**Why the model of Appendix A.1 could not see this effect.** Its spectrum has two atoms and **no mass near
$\eta_j/\varepsilon=1$**, where the entire effect lives; relaxing its threshold changes nothing until
the whole bulk returns at once and robustness collapses 22 points. That model rules out the paper's
effect by construction — which is exactly why it finds anchoring and plain AT to be the same
classifier, and why A.1 explains only the *free* half.

⚠ **The limit of every re-weighting account.** For a linear readout
$M_c(c)-M_r(c)=\varepsilon\lVert c\rVert_1/\lVert c\rVert_2$ *exactly*: reading a coordinate and being
exposed on it are the same parameter, so deletion is the only gap-reducing move available, and across
four spectra × six radii the anchored objective's own optimum trades at 0.46–2.6 against a measured
rate $\ge12$. **A nonlinear map escapes this** because its exposure is $O$, which is not a function of
how many coordinates it reads: it can depend on an attackable coordinate and still be locally
constant in it. That is what §3's $F+O$ asks for and adversarial CE does not.

---

## 6. Results

AA is AutoAttack (`apgd-ce`+`apgd-t`) on all 10 000 test images at $\varepsilon=8/255$; NRR is the
harmonic mean of clean and AA.

**CIFAR-100 / ResNet-18** — full comparison in §1.

| | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **Ours** | **62.17** | 34.77 | **30.92** | **28.59** | **39.17** |
| ADR + WA + AWP | 57.36 | 34.92 | 30.62 | 28.50 | 38.08 |

**CIFAR-10 / ResNet-18** — identical configuration, differing only in dataset and teacher checkpoint.

| | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **Ours** | **84.66** | 56.74 | **53.94** | 51.87 | **64.33** |
| ADR + WA + AWP | 83.26 | — | — | 51.18 | 63.39 |
| CURE | 86.76 | 54.92 | 52.48 | 49.69 | 63.19 |

**Tiny-ImageNet-200 / ResNet-18** — same recipe transferred unchanged.

| | clean | AA | NRR |
|---|---:|---:|---:|
| **Ours** (200ep teacher) | **55.16** | **20.54** | **29.93** |
| Ours (80ep teacher) | 57.08 | 18.96 | 28.46 |
| ADR + WA + AWP | 48.27 | 20.10 | 28.38 |

Tiny-ImageNet with the longer teacher wins **both** axes outright (+6.89 clean, +0.44 AA).
Warm-starting from the teacher rather than from scratch is worth **+2.02 AA**, so the initialization
is part of the method and not an implementation detail.

*Protocol.* Our runs and the ADR rows both use WA and AWP. ADR trains 80 epochs on Tiny-ImageNet and
we train 100; our Tiny-ImageNet teachers are themselves short (65.97 / 66.29 clean) and had not
converged.

---

## 7. Ablations

**Logit anchoring, priced.** Same teacher, same regime, same attack, no interventions — only $\tau$.
Base regime: raw features, 50 epochs, no WA, no AWP, $\varepsilon=8/255$.

| | clean | PGD-20 | CW | **AA** | **NRR** |
|---|---:|---:|---:|---:|---:|
| logits, $\tau=1$ | 58.26 | 22.62 | 22.79 | 20.84 | 30.70 |
| logits, **$\tau=4$** | 59.39 | 30.30 | 26.84 | **24.48** | **34.67** |
| logits, $\tau=16$ | 57.78 | 31.57 | 26.11 | 24.00 | 33.91 |
| feature anchor + head KD, $\tau=1$ | 62.34 | 26.34 | 26.11 | 23.93 | 34.58 |
| feature anchor + head KD, $\tau=16$ | 62.61 | **32.47** | 27.32 | 25.61 | 36.35 |
| **feature anchor, no head KD** | **62.72** | 28.69 | **27.80** | **25.88** | **36.64** |
| ‥ + head refit, adversarial CE | 62.77 | 29.93 | 27.66 | 25.65 | 36.42 |
| ‥ + head refit, adv CE + smoothing 0.1 | 62.57 | 30.43 | 27.55 | 25.66 | 36.39 |
| ‥ + head refit, clean CE | 62.70 | 28.90 | 27.23 | 25.15 | 35.90 |

**The feature anchor beats the best temperature on every axis** (+3.33 clean, +0.96 CW, +1.40 AA,
+1.97 NRR) with no temperature to have chosen. The logit curve is $\cap$-shaped because $\tau$ is
squeezed from both sides: at $\tau=1$ the teacher sits at 0.820 max-prob against the student's 0.016,
so the student is asked for a confidence 20× sharper than its own; at $\tau=16$ the target is at
0.0196, within 1% of uniform, *and* the warm-started student begins 16-fold away from it, collapsing
to clean 1.01 in the first epoch. **$\tau$ has to be large and small at once.**

**The head term is worth removing, not tuning.** No temperature beats deleting it (36.35 at the best
$\tau$ against 36.64 with the term gone), and every form of head re-training lands below leaving the
teacher's head untouched. Label smoothing lands where hard labels do, so "hard labels are too sharp"
is not the explanation. Same verdict at the full recipe: deleting costs $-0.06$ AA, misconfiguring
costs $-0.91$.

**Sensitivity-matched $\varepsilon$, and the signal is what matters.** At the full recipe $p{=}1$ is
worth +1.43 clean at $-0.10$ AA on CIFAR-100 and +2.14 at $-0.02$ on CIFAR-10, at identical total
attack budget. Keeping the *exact* weight multiset the sensitivity rule produces and only reassigning
which sample gets which — ordering by difficulty, as IAAT/CAT do — buys clean and **pays 0.74 AA**,
ending below the uniform baseline on both CW and NRR. Only the sensitivity ordering moves the
frontier rather than sliding along it. ⚠ A control, not a comparison against those methods as
published; they have not been run.

**PGD would have inverted most of this table.** Head KD at $\tau=16$ has the best PGD-20 (32.47) and a
worse AA than the row with no head term (25.61 against 25.88); across the smoothing sweep PGD rises
monotonically while CW falls. Only AutoAttack arbitrates.

---

## 8. What is not claimed

- **That a natural teacher beats a robust one.** Our comparison class is methods without a robust
  teacher. Where a robust teacher of comparable capacity is available, expect it to do at least as
  well; our results say the *anchor term* gains nothing from it.
- **That AA improves on CIFAR.** It does not — those gains are on the clean axis at matched
  robustness. Tiny-ImageNet with a 200ep teacher is the one configuration that gains on both.
- **That the directional and raw targets differ.** Normalizing both sides is a tie: at each design's
  own best schedule, NRR 39.17 against 39.29, inside noise, reproduced on a second machine.
- **That the model of §5.2 predicts the size of the effect.** It predicts the *shape* — a discontinuous
  threshold and a diverging exchange rate near it — and its own optimum trades at 0.46–2.6 against a
  measured $\ge12$.
- **Anything about why WA, AWP or the learning-rate schedule help.** They are not part of the method
  and we have no account of them.
- **That removing the teacher's attack-volatile directions helps.** It cannot be done: the top-100
  volatile directions carry 92.8% of the attack displacement and 96.6% of the clean feature energy.

## 9. Open

**Related work.** The claim that feature-vs-logit is a tuning decision in ordinary KD and a
structural one here rests on our gap being far larger than the usual one. Establishing the usual one
is a prerequisite, not a nicety.

**Saturation.** AA flattens after a 150ep teacher while $DL/\gamma_t$ keeps falling; the empirical
rotation-over-class-gap ratio flattens with it. Whether that is Cauchy–Schwarz slack or a sign that
the right denominator is the class-mean gap is unresolved.

**The Tiny-ImageNet ladder has not peaked.** NRR is still rising at the longest teacher trained
(28.46 → 28.57 → 29.93), so the optimum there is above 200 epochs and unmeasured.

---

## Appendix A. Two partial results on the robustness side

§3 explains why anchoring a non-robust teacher is *free*. Neither result below closes the
complementary question — how much robustness the objective buys — and both are reported with the
reason they fall short, because each is instructive about where a proof would have to come from.

### A.1 In the two-feature model, the non-robust coefficient is zeroed rather than shrunk

Tsipras et al.: robust feature $x_1=y$ w.p. $p$, non-robust $z_j\sim\mathcal N(\eta y,1)$, an
$\ell_\infty$ adversary on $z$ only, natural teacher $\Phi_t=m(z)$. Take $\Phi_s=ax_1+b\,m(z)$.

**Proposition A1.** With the inner maximum exact,
$J(a,b)=\mathbb{E}[R^2]+2\varepsilon\lvert b\rvert\mathbb{E}\lvert R\rvert+\varepsilon^2b^2$ with
$R=ax_1+(b-1)m$, kinked at $b=0$ with subgradient jump $2\varepsilon\mathbb{E}\lvert R(a,0)\rvert>0$;
above a threshold $\varepsilon_0$ the minimizer has $b^\star=0$ **exactly**.

The adversarial term is an $\ell_1$ penalty on the non-robust coefficient: the teacher supplies the
value, the inner maximization discards the route it took there. Verified numerically ($d=512$):
$b^\star=0.510$ at $0.5\eta$, exactly $0$ at $\eta,2\eta,4\eta$. Since every $\varepsilon$ in $J$
multiplies $b$, at $b^\star=0$ the objective is $\mathbb{E}(ax_1-m)^2$ and is $\varepsilon$-free —
past the threshold a larger radius costs the anchor nothing, which matches the measured saturation
(raising $\varepsilon_{\mathrm{tr}}$ from 8.8 to 10/255 on Tiny-ImageNet leaves the clean penalty flat
at $-2.4$ for the whole run rather than widening).

⚠ **This model rules out the paper's effect by construction**, so A1 cannot be the explanation of it.
Its reliability spectrum has two atoms and **no mass near $\eta_j/\varepsilon=1$**, which §5.2 shows
is where the entire effect lives; on that spectrum the anchored optimum and plain AT are the *same
classifier*, and relaxing the threshold changes nothing until the whole bulk returns at once and
robustness collapses 22 points. A1 is a statement about the free half only, and its own model is a
counterexample to using it for more.

### A.2 What robust accuracy tracks is the loss measured against a margin

Because the head is frozen at the teacher's, Cauchy–Schwarz gives a per-sample certificate.

**Proposition A2.** With $\gamma_t(x)=\min_{c\ne y}\langle w_y-w_c,\Phi_t(x)\rangle$ the teacher's
clean margin and $D_y=\max_{c\ne y}\lVert w_y-w_c\rVert$: if $\gamma_t(x)>D_yL(x)$ then $x$ is
robustly correct, since $\langle w_y-w_c,\Phi_s(x')\rangle\ge\gamma_t(x)-D_yL(x)$ on the whole ball.

The quantity it names is the one that behaves. Across the teacher ladder,

$$r\big(DL/\gamma_t,\ \mathrm{AA}\big)=-0.971,\qquad r\big(DL/\gamma_t,\ \text{clean}\big)=+0.977,$$

both axes at once and with the sign that $O$ alone gets wrong. $\mathbb{E}[\gamma_t]$ is flat along
the ladder (4.20→4.57), so the ratio moves because $L$ falls: **longer-trained teachers are easier to
match under attack**, which is the mechanism behind §4.

⚠ **The certificate is vacuous** — 1.5–7.9% certified against a measured AA of 24–26. Cauchy–Schwarz
assumes the displacement aligns with $w_y-w_c$, which does not happen, and $\mathbb{E}[L]\approx7.8$
against $\mathbb{E}[\gamma_t]\approx4.5$ means it cannot bite on the average sample. It also misses
the saturation: AA flattens after the 150ep teacher while $DL/\gamma_t$ keeps falling, whereas the
empirical rotation-over-class-gap ratio flattens with it ($r=-0.991$). **What survives is the
quantity, not the guarantee.**

