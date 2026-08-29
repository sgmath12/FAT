# Theory section (paper draft, 2026-08-08)

Formal statements only. Every result is followed by a **Prediction** line naming the experiment that
tests it; results with no testable consequence are not in this section. `method_v2.md` remains the
lab notebook — this file is what goes in the paper.

**Roadmap (restructured 2026-08-21).** The method has three parts and each has exactly one theorem
behind it:

| contribution | what it claims | theory |
|---|---|---|
| anchor to a **natural** teacher | the anchor is a robustness-free knob, so a cheap clean checkpoint is a usable guide | **T.1**, Thm 1 + Cor. 1 |
| **separate** the head from the backbone | the teacher's head is not optimal for the student's feature, so it must be re-solved rather than inherited | **T.2**, Thm 2 |
| **sensitivity-matched ε** (angeps) | a fixed pixel radius moves different samples by wildly different amounts of the quantity the loss measures; equalize that instead, at fixed total budget | **T.5** |

⚠ **What is deliberately NOT a contribution.** Whether the backbone target is the *direction*
($\hat\Phi_s$ vs $\hat\Phi_t$) or the *raw* feature ($\Phi_s$ vs $\Phi_t$) is **a tie**: given each
design its own best schedule, NRR 39.17 vs 39.29 on CIFAR-100 (RESULTS.md §5e), inside the noise
band. T.4 below is kept only for the structural facts it establishes (Prop. 4 and the 2×2 algebra);
its former conclusion — that the raw target is worse, and that this is "the argument for the design"
— does not survive and must not be quoted.

---

## T.0 Setup

Student backbone $\Phi_s$, head $W_s\in\mathbb{R}^{K\times d}$ reading the **normalized** feature,
$z_s(x)=W_s\hat\Phi_s(x)+b_s$. Frozen teacher $(\Phi_t,W_t)$, trained naturally. Objective (2):

$$\mathcal{L}=\underbrace{\mathbb{E}\big\lVert\hat\Phi_s(x_{\mathrm{adv}})-\hat\Phi_t(x)\big\rVert^2}_{\mathcal{L}_{\mathrm{dir}}\ \to\ \theta_{\mathrm{bb}}}
+\beta\underbrace{\mathbb{E}\,\mathrm{KL}\big(\mathrm{sm}(z_t/\tau)\,\Vert\,\mathrm{sm}(W_s\,\mathrm{sg}[\hat\Phi_s(x_{\mathrm{adv}})]+b_s)\big)}_{\mathcal{L}_{\mathrm{hd}}\ \to\ \theta_{\mathrm{hd}}}$$

The backbone term is a plain cosine loss,
$\mathcal{L}_{\mathrm{dir}}=\mathbb{E}[\,2-2\cos\angle(\Phi_s(x_{\mathrm{adv}}),\Phi_t(x))\,]$, and the
inner maximization is the PGD that opens that angle. *(The implementation carries an optional
orthonormal $Q\in\mathbb{R}^{d\times k}$ projecting the loss and the attack onto a $k$-dimensional
subspace. The shipped model uses $k=d$, where $Q$ is square orthogonal and
$\lVert Q^\top v\rVert=\lVert v\rVert$ — an exact no-op. It is a knob left over from a retired
subspace ablation and is therefore absent from every statement below.)*

**Proposition 0 (routing).** *With the stop-gradient, $\partial\mathcal{L}/\partial\theta_{\mathrm{bb}}=\partial\mathcal{L}_{\mathrm{dir}}/\partial\theta_{\mathrm{bb}}$ and $\partial\mathcal{L}/\partial\theta_{\mathrm{hd}}=\beta\,\partial\mathcal{L}_{\mathrm{hd}}/\partial\theta_{\mathrm{hd}}$.*

*Proof.* $\mathcal{L}_{\mathrm{dir}}$ contains no head parameter; in $\mathcal{L}_{\mathrm{hd}}$ the
feature enters only through $\mathrm{sg}[\cdot]$, whose Jacobian in $\theta_{\mathrm{bb}}$ is zero. $\square$

Prop. 0 is not decoration: it is what makes the fixed-head hypothesis of **Theorem 3** hold exactly
on our side of the comparison.

---

## T.1 What anchoring to a *natural* teacher selects

**Model (Tsipras et al.).** $y\sim\mathrm{Unif}\{\pm1\}$; $x_1=y$ w.p. $p=0.95$, else $-y$;
$z_1..z_d\stackrel{iid}\sim\mathcal{N}(\eta y,1)$ with $\eta=4/\sqrt d$; $\ell_\infty$ adversary of
radius $\varepsilon$ acting on $z$ only ($x_1$ unperturbable). Write $q:=2p-1=0.9$,
$v:=1-q^2=0.19$, $m(z):=\frac1d\sum_j z_j$. The natural teacher is $\Phi_t=m(z)$
(standard accuracy $\Phi(4)\approx99.99\%$, robust accuracy $\approx0$).

**Assumption T.1.** The student is linear in the two feature groups,
$\Phi_s=a\,x_1+b\,m(z)$, $(a,b)\in\mathbb{R}^2$.

**Theorem 1 (exact suppression of the non-robust feature).**
*Let $J(a,b)=\mathbb{E}\max_{\lVert\delta\rVert_\infty\le\varepsilon}\big(\Phi_s(x+\delta)-\Phi_t(x)\big)^2$.
Then*

$$J(a,b)=\mathbb{E}\big[R^2\big]+2\varepsilon\lvert b\rvert\,\mathbb{E}\lvert R\rvert+\varepsilon^2b^2,
\qquad R:=a\,x_1+(b-1)\,m(z),$$

*and $J$ is non-differentiable at $b=0$ with a strictly positive subgradient jump
$2\varepsilon\,\mathbb{E}\lvert R(a,0)\rvert$. Consequently there is a threshold
$\varepsilon_0$ such that for all $\varepsilon\ge\varepsilon_0$ the minimizer has*

$$b^\star=0\quad\text{exactly},\qquad a^\star=q\,\eta .$$

*Proof.* The adversary controls $m$ through $\bar\delta=\frac1d\sum\delta_j\in[-\varepsilon,\varepsilon]$
and maximizes $\lvert R+b\bar\delta\rvert$, so the inner max equals
$(\lvert R\rvert+\varepsilon\lvert b\rvert)^2$; expanding gives the display. The $\lvert b\rvert$ term
is **linear** in $\lvert b\rvert$ at the origin with slope $2\varepsilon\mathbb{E}\lvert R(a,0)\rvert>0$,
whereas the only terms that reward $b\neq0$ — $\mathbb{E}R^2$ through $(b-1)$ — are quadratic. A
kinked linear penalty against a quadratic gain yields exact sparsity once the slope dominates, which
is the stated threshold. Given $b^\star=0$, $J(a,0)=\mathbb{E}(ax_1-m)^2$ is minimized at
$a^\star=\mathbb{E}[x_1m]/\mathbb{E}[x_1^2]=q\eta$. $\square$

> **The adversarial term is an $\ell_1$ penalty on the non-robust coefficient.** This is the whole
> mechanism, and it is why a natural teacher is enough: the teacher supplies the *value* to be
> matched, and the inner maximization supplies the sparsity that discards the route the teacher took
> to it.

> **What Theorem 1 is for — and the trap in stating it.** Guiding adversarial training with a
> clean-trained network is established practice, so *"a natural teacher is usable"* is not a finding
> and must not be presented as one. The standing **objection** is what has content: that a
> non-robust anchor imports the teacher's non-robust features and therefore **taxes robustness** —
> the expectation under which ADR self-anchors and the ARD/RSLAD line requires a robust teacher.
> Theorem 1 answers that objection **inside the model where the objection was formulated**
> (Tsipras et al.), which is the only sense in which it is worth stating: robustness is set by the
> inner maximization alone, and the anchor contributes only the value to be matched.
>
> **Be explicit about what the model gives away.** It assumes $x_1$ is literally unperturbable and
> $z$ fully perturbable, so *some* suppression of $z$ is built in before anything is proved — a
> referee will say so, and should. What is **not** built in, and is the whole of the theorem's
> content, is three things:
> 1. the suppression is **exact** ($b^\star=0$, not asymptotically small) — a kink/$\ell_1$
>    phenomenon, not a generic consequence of adding an adversary;
> 2. it is **conditional on a threshold** $\varepsilon_0$ — below it the non-robust feature is
>    *retained* ($b^\star=0.518$ at $0.5\eta$), so the claim is falsifiable rather than automatic;
> 3. past the threshold the anchored solution is **$\varepsilon$-independent** (Cor. 2 below), which
>    is what separates this objective from label-driven AT and is directly measurable.
>
> Stated that way the section claims a *mechanism and its condition*, not the practice.

**Numerically** ($d=512$, Monte-Carlo; re-verified independently 2026-08-22, $n=2\cdot10^5$):

| $\varepsilon$ | $b^\star$ (exact) | $b^\star_{\mathrm{quad}}$ (kink dropped) |
|---|---:|---:|
| $0.5\eta$ | 0.510 | 0.502 |
| $1.0\eta$ | **0.000** | 0.202 |
| $2.0\eta$ (the model's radius) | **0.000** | 0.060 |
| $4.0\eta$ | **0.000** | 0.016 |

so $\varepsilon_0\in(0.5\eta,\,1.0\eta)$ and the shipped regime sits far above it. **The theorem's
own content — $b^\star=0$ exactly for every $\varepsilon\ge\eta$ — reproduces.**

Dropping the $\lvert b\rvert$ kink leaves $J_{\mathrm{quad}}=\mathbb{E}[R^2]+\varepsilon^2b^2$.
Writing $u=1-b$ and minimizing out $a$ (which gives $a=q\eta u$) collapses it to
$u^2(\eta^2v+1/d)+\varepsilon^2(1-u)^2$, hence

$$b^\star_{\mathrm{quad}}=\frac{c}{c+\varepsilon^2},\qquad c=\eta^2v+\tfrac1d,$$

which matches the numerical column above to three decimals.

⚠ **Two corrections to the previous version of this paragraph (2026-08-22).** It printed
$c_1=\eta^2v/q^2+1/d$; the $1/q^2$ is spurious — the $q^2$ cancels when $a$ is minimized out — and it
inflated the column to $0.543/0.229/0.069/0.018$. And it called the relaxation *"a rigorous upper
bound; the exact answer is stronger"*, which is false: at $0.5\eta$ the exact minimizer $0.510$
**exceeds** the relaxed one $0.502$. The monotonicity argument behind "upper bound" does not apply,
because the dropped term $2\varepsilon\lvert b\rvert\mathbb{E}\lvert R\rvert$ is not a pure penalty
in $b$ — $\mathbb{E}\lvert R\rvert$ depends on $a$ as well, so removing it also moves $a^\star$. The
relaxation is a convenient closed form, not a bound, and should be presented as such.

**Corollary 2 ($\varepsilon$-independence past the threshold).** *For $\varepsilon\ge\varepsilon_0$
the minimizer of $J$ does not depend on $\varepsilon$ at all: $b^\star=0$ and $a^\star=q\eta$ for
every larger radius.*

*Proof.* Every $\varepsilon$ in $J(a,b)=\mathbb{E}[R^2]+2\varepsilon\lvert b\rvert\mathbb{E}\lvert R\rvert+\varepsilon^2b^2$
multiplies $b$. At $b=0$ the objective collapses to $J(a,0)=\mathbb{E}(ax_1-m)^2$, in which
$\varepsilon$ does not appear. $\square$

This is the qualitative difference from label-driven AT, whose CE objective keeps deforming the
decision boundary as $\varepsilon$ grows, so clean accuracy falls monotonically. Here $\varepsilon$
decides one thing — whether the non-robust route is cut — and once it is cut, a larger radius costs
the anchor nothing. **Prediction 1c: clean accuracy should *saturate* in $\varepsilon_{\mathrm{tr}}$
past the threshold rather than decline steadily.**

⚠ Measured, the cost is small but not zero: $\varepsilon_{\mathrm{tr}}$ $8\to8.8$ costs clean
$-1.38$ (direction) / $-1.49$ (L2) and buys AA $+0.75$ / $+0.45$. The corollary is about the
**anchor term**; the residual presumably belongs to the head-KD term and to finite capacity. So the
claim to make is *saturation*, not *free* — and it needs a sweep with more than the two radii we
have (an $\varepsilon_{\mathrm{tr}}$ curve at fixed recipe, currently unrun).

**Prediction 1a.** Enlarging $\varepsilon_{\mathrm{tr}}$ past the threshold suppresses non-robust
features; below it the student keeps them. → the $\varepsilon_{\mathrm{tr}}$ sweep, and the shipped
$\varepsilon_{\mathrm{tr}}=1.1\varepsilon$ choice.

**Corollary 1 (in the model, at matched teacher capacity, the teacher's robustness is second-order).**
*Replacing $\Phi_t=m(z)$ with a robust teacher $\Phi_t=\eta x_1$ gives $b^\star=0$ and
$a^\star=\eta$, i.e. the same support and the same feature, differing only by the scalar gain
$q=0.9$.*

*Proof.* $J=\mathbb{E}[(\lvert(a-\eta)x_1+bm\rvert+\varepsilon\lvert b\rvert)^2]$, minimized at
$(a,b)=(\eta,0)$ with $J^\star=0$. $\square$

The two teachers select **the same student feature**; the entire difference is a multiplicative gain,
which is exactly the quantity a re-solved head absorbs (Theorem 2).

**Scope — what this does and does not license.** It licenses the *cheap* teacher: at matched teacher
capacity, the anchor gains nothing from the teacher being robust, so the natural checkpoint is not a
compromise *within this setting*. The claim to make is the modest one — **a natural teacher is a
usable guide** — and not a comparative one. Where a robust teacher of comparable capacity is
available, one should expect it to do at least as well; the model says the *anchor* gains nothing
from it, which is a statement about this one term, not about the methods that spend their budget on
a stronger teacher. It must **not** be stated as "a robust teacher is unnecessary" —
that reads as a ranking claim against the robust-teacher KD line (ARD, RSLAD, …), whose numbers are
bought substantially from **stronger** teachers (large robust WRNs), an axis the model holds fixed
and this paper does not compete on. The comparison class for our results is methods *without* a
robust teacher (ADR, TRADES-family), and the paper must say so explicitly.

**Prediction 1b.** Swapping the natural teacher for a **same-architecture** robust one changes the
result by ≈0. → the robust-teacher ablation (checkpoints already on disk). This is a *mechanism
diagnostic*, not a superiority claim.

**Remark (scope).** Scalar features, linear student, $x_1$ literally unperturbable. The model
predicts *which feature group survives*, not accuracies. Note also that Theorem 1 nowhere uses that
$\Phi_t$ is a **feature**: it uses only that the anchor is computed by a frozen network on the clean
input and is a proxy for $y$. What Theorem 1 therefore does *not* explain is why the anchor should
live in feature space; that is Theorem 3's subject, and it is a clean-axis argument.

---

## T.2 The head cannot be inherited

⚠ **Demoted to a remark, and its arithmetic corrected (2026-08-22).** "If the student's feature
differs from the teacher's, the teacher's optimal head is not the student's" is close to
definitional — the feature space changes, so of course the classifier is refit. Presenting it as a
theorem weakens the section. What is **not** definitional, and is where the content moved, is
**Prop. 0**: the head's objective must be routed away from the backbone entirely
(`featdir_alpha` $=0$). The obvious alternative — letting the head KD train the backbone too, which
is how such a loss is usually written — is a real design choice with a real ablation
(`featdir_alpha1_champion`).

**Remark (feature statistics).** *Under Theorem 1's solution $a^\star=q\eta$, so*

| | mean | variance |
|---|---|---|
| teacher $\Phi_t=m(z)$ | $\eta y$ | $1/d$ |
| student $a^\star x_1$ | $q^2\eta y=0.81\,\eta y$ | $q^2\eta^2v=\mathbf{2.46}/d$ |

*with $q=0.9$, $v=1-q^2=0.19$, $\eta^2=16/d$: gain ratio $q^2=0.81$, noise ratio $2.46$.*

*(The earlier version of this table read mean $q\eta y$ and variance $3.04/d$. Those are
inconsistent with each other — $3.04/d=\eta^2v$ drops the $q^2$ that the mean row keeps — and both
disagree with $a^\star=q\eta$. Since $\mathbb{E}[x_1\mid y]=qy$, the mean is
$a^\star\cdot qy=q^2\eta y$ and the variance is $a^{\star2}v=q^2\eta^2v$.)*

Away from the scalar model, write the adversarial feature in the teacher's frame with
$\rho=\cos\angle(\Phi_s(x_{\mathrm{adv}}),\Phi_t(x))$:
$\hat v=\rho\hat u+\sqrt{1-\rho^2}\,e$, so
$\langle w_c,\hat v\rangle=\rho\langle w_c,\hat u\rangle+\sqrt{1-\rho^2}\langle w_c,e\rangle$.
$w_t$ is the $\rho=1$ solution and the mismatch ratio is $R=\sqrt{1-\rho^2}/\rho$, measured at
$0.77$ ($\varepsilon_{\mathrm{tr}}=8/255$) and $1.04$ (champion, $8.8/255$).

**Prediction 2.** The cost of freezing the head grows **monotonically** in $\varepsilon_{\mathrm{tr}}$
(a curve, not a sign test), because $\rho$ falls and $R$ grows. → head-freeze × $\varepsilon_{\mathrm{tr}}$ grid, 3 points.

⚠ **Measured caveat (2026-08-08, $n=10^4$).** The contamination term is *not* near worst case:
the fraction of the feature displacement that converts into margin loss is $0.22$, against $1.0$ for
a worst-case residual and $1/\sqrt d=0.044$ for a random one — and a head-*aware* CE adversary
reaches only $0.24$ versus $0.22$ for the head-*blind* feature attack. So $e$ is adversarial but far
from aligned, and the earlier claim that $\langle w_c,e\rangle$ "cannot be dismissed" overstates the
size. Theorem 2's *direction* survives; its magnitude must be quoted from the measurement.

---

## T.2b What the adversarial feature loss is, exactly

The only statement in this file that needs no model, no linearity and no assumption beyond the
triangle inequality — and the one that says what the objective actually asks for.

Fix $x$, write $B(x)$ for the $\varepsilon$-ball around it, and define

$$L(x)=\max_{x'\in B(x)}\lVert\Phi_s(x')-\Phi_t(x)\rVert,\qquad
F(x)=\lVert\Phi_s(x)-\Phi_t(x)\rVert,\qquad
O(x)=\!\!\max_{x',x''\in B(x)}\!\!\lVert\Phi_s(x')-\Phi_s(x'')\rVert .$$

$L$ is the training loss (before squaring, with the inner maximum exact); $F$ is **fidelity** — how
well the student reproduces the teacher at the clean point; $O$ is the **oscillation** of the
student's own feature map over the ball, which is what it means for a feature map to be robust.

**Proposition 2.** $\;L\le F+O\le 3L.$

*Proof.* $x\in B(x)$ gives $F\le L$. For any $x',x''\in B(x)$,
$\lVert\Phi_s(x')-\Phi_s(x'')\rVert\le\lVert\Phi_s(x')-\Phi_t(x)\rVert+\lVert\Phi_t(x)-\Phi_s(x'')\rVert\le 2L$,
so $O\le 2L$ and hence $F+O\le 3L$. Conversely, for any $x'\in B(x)$,
$\lVert\Phi_s(x')-\Phi_t(x)\rVert\le\lVert\Phi_s(x')-\Phi_s(x)\rVert+\lVert\Phi_s(x)-\Phi_t(x)\rVert\le O+F$;
maximizing over $x'$ gives $L\le F+O$. $\square$

So the single term is equivalent, within a factor of three, to **the sum of two requirements**:
reproduce the teacher on clean inputs, *and* be constant on $\varepsilon$-balls. Driving it down
drives both down. Nothing else in the objective is needed to obtain either.

**This is the answer to "why is a non-robust teacher a usable target".** $\Phi_t$ is evaluated at
$x$ and nowhere else, so the teacher's own instability never enters. The student is not asked to
imitate the teacher; it is asked to hold the teacher's clean value while under attack — which the
teacher cannot do, and which is the entire content of $O$.

**Contrast with the logit route.** The same decomposition applies to
$\mathrm{KL}(\mathrm{sm}(z_t(x)/\tau)\Vert\mathrm{sm}(z_s(x_{\mathrm{adv}})))$, but its fidelity
term is fidelity to a *temperature-distorted* teacher rather than to the teacher, and $\tau$ enters
there. The feature route has no such distortion to calibrate.

**Measured (2026-08-23, train split, each cell's own matched attack).** The bound is not vacuous and
the two terms are of comparable size, with fidelity the larger:

| | $L$ | $F$ | $O_{\text{lb}}$ | $F+O_{\text{lb}}$ | $3L$ | teacher's own $O$ |
|---|---:|---:|---:|---:|---:|---:|
| champion (directional) | 0.674 | 0.542 | 0.238 | 0.780 | 2.022 | **0.566** |
| L2 at its own best recipe | 7.583 | 6.304 | 2.164 | 8.469 | 22.749 | **6.684** |

The student oscillates **2.4× less than the teacher** on the same balls (0.238 against 0.566), which
is $O$ doing its job, and matches the independent angular measurement (12–13° against 62°). Most of
what remains in the loss is fidelity, not instability.

⚠ Three caveats to carry. $O$ is only *lower*-bounded here, by
$\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_s(x)\rVert$, since the true maximum over pairs is its own
optimization. $L$ likewise is what PGD reaches, a lower bound on the exact inner maximum, while the
proposition is stated for the exact one. And the proposition says nothing about whether small $F$ is
*desirable* — that rests on the teacher's clean features being a good representation, which is an
assumption supported by its 77.4% clean accuracy, not by the inequality.

*Scope.* $O$ is the oscillation of the **feature map**. Robust *classification* additionally needs a
head that does not undo it, which is a separate matter and is why the head is fitted rather than
inherited in the shipped recipe.

---

## T.2c What the feature anchor has that a logit anchor does not

Read as a continuation of T.2b, not as a criticism of logit distillation — which works well in the
robust-teacher line and is not what is being argued against. The question is narrower: given a
*natural* teacher and a warm-started student, which of the properties T.2b establishes survive when
the anchor is moved to logits.

**1. The decomposition needs the loss to be a metric, and the feature loss is one.**
Proposition 2 is proved by the triangle inequality on $\lVert\cdot\rVert$, which is what lets a
single term be split into fidelity $F$ and oscillation $O$. The logit objective is a KL divergence,
which is not symmetric and does not obey a triangle inequality, so there is no corresponding
$L \le F + O \le 3L$ to be had — the guarantee is a property of anchoring in a metric space, not of
anchoring in general. *(A metric surrogate such as total variation or Hellinger would restore it,
but that is a different objective from the one the KD literature uses.)*

**2. The feature target is used as it is; a logit target is used through a projection and a
saturating map.** With $z_t = W_t\Phi_t$, $W_t \in \mathbb{R}^{K\times d}$, $K=100$, $d=512$, the
student is told 100 numbers rather than 512, and then only through $\mathrm{sm}(\cdot/\tau)$.
*(This counts the target's own coordinates. It is not the rank argument retracted in T.3, which
wrongly counted feature directions against a loss living in logit space.)*

**3. There is no temperature to choose, because $\tau$ is squeezed from both sides.** Two exact
statements about $p_\tau = \mathrm{sm}(z_t/\tau)$:

*Small $\tau$ — saturation.* With a unique maximizer $p_\tau \to \mathrm{onehot}(\arg\max z_t)$, so
in the limit the target depends on $z_t$ **only through its argmax**. This is destruction, not
attenuation: no loss weight recovers it.

*Large $\tau$ — flattening.* Expanding $\exp(z_i/\tau)=1+z_i/\tau+O(\tau^{-2})$,

$$(p_\tau)_i=\frac1K+\frac{z_i-\bar z}{K\tau}+O(\tau^{-2}),\qquad \bar z=\tfrac1K\textstyle\sum_j z_j,$$

so the teacher's structure survives only at order $1/\tau$; the whole target lies within
$O(1/\tau)$ of uniform. This is attenuation and could in principle be met by raising $\beta$.

*Checked against the teacher* ($\max_i z_i-\bar z = 10.35$, so the prediction is
$0.01+0.1035/\tau$):

| $\tau$ | measured $\max_i (p_\tau)_i$ | first-order prediction |
|---|---:|---:|
| 1 | 0.8212 | 0.1135 |
| 4 | 0.1527 | 0.0359 |
| 16 | **0.0196** | **0.0165** |
| 32 | 0.0139 | 0.0132 |

The expansion is accurate from $\tau\approx16$ and fails below it — the failure region being exactly
the saturation regime above. Uniform is $0.0100$: at $\tau=16$ the target is already within $1\%$ of
carrying nothing.

**4. The feature objective is exactly zero at initialization; a logit objective is not, except
where it is empty.** The student starts as the teacher, so $\Phi_s=\Phi_t$ and Proposition 2's $F$
vanishes before training — only the perturbation makes the objective non-zero, which is precisely
the quantity one wants to train. A logit target is zero at initialization only at $\tau=1$, which is
where saturation has emptied it. Measured at $\tau=16$ with no scale correction, clean accuracy
falls to **1.01%** in the first epoch and the run is still recovering fifty epochs later.

**Summary.** Statements 1–4 are properties the feature anchor has: a decomposition into fidelity and
stability, an undistorted target, no temperature to select, and a starting point that already
satisfies the fidelity half. Points 3 and 4 are specific to a *natural* teacher — its confidence is
what makes $\tau$ necessary and awkward — and a robust teacher, being far less confident, is subject
to them much more weakly. **Nothing here argues that logit distillation is a poor method; it argues
that these four properties are why the anchor is placed on features in this setting.**

---

## T.2d What the loss has to be small *relative to*

Proposition 2 says the loss controls fidelity $F$ and oscillation $O$. It does not say either
controls **accuracy**, and §9.3 of `METHOD.md` shows why that gap matters: across the teacher ladder
$O$ correlates with AA at $r=+0.83$ — the *wrong sign*. Teachers that oscillate more produce
students that are more robust. Oscillation alone is not a sufficient statistic for robust accuracy;
it has to be priced against the margin it must cross. This section derives that pricing rather than
observing it.

**Setup.** The head is frozen at the teacher's, $w=w_t$ (§2). For a sample $(x,y)$ write the
teacher's clean margin and the head's class-pair diameter

$$\gamma_t(x)=\min_{c\ne y}\langle w_y-w_c,\ \Phi_t(x)\rangle,\qquad
D_y=\max_{c\ne y}\lVert w_y-w_c\rVert .$$

**Proposition 3 (margin certificate).** *If $\gamma_t(x) > D_y\,L(x)$ then $x$ is robustly correct:
the student's prediction is $y$ for every $x'\in B(x,\varepsilon)$.*

*Proof.* For any $x'\in B(x)$ and any $c\ne y$,

$$\langle w_y-w_c,\Phi_s(x')\rangle
=\underbrace{\langle w_y-w_c,\Phi_t(x)\rangle}_{\ \ge\ \gamma_t(x)}
+\langle w_y-w_c,\ \Phi_s(x')-\Phi_t(x)\rangle
\ \ge\ \gamma_t(x)-\lVert w_y-w_c\rVert\,\lVert\Phi_s(x')-\Phi_t(x)\rVert$$

by Cauchy–Schwarz, and $\lVert\Phi_s(x')-\Phi_t(x)\rVert\le L(x)$ by definition of $L$ as the inner
maximum. So every margin stays positive whenever $\gamma_t(x)>D_yL(x)$. $\square$

Again nothing is assumed about the network — Cauchy–Schwarz and the definition of $L$. But unlike
Proposition 2 this one names the quantity accuracy depends on: **not $L$, but $L$ measured against
the teacher's own margin.** It also explains structurally why the teacher matters beyond supplying a
target: $\gamma_t$ is a property of the *teacher*, and the teacher ladder moves it.

**Measured (CIFAR-100, teacher ladder, 12 test batches per cell, each cell's own matched attack).**

| teacher | certified | median $D L/\gamma_t$ | $\mathbb{E}[\gamma_t]$ | $\mathbb{E}[L]$ | AA | clean |
|---|---:|---:|---:|---:|---:|---:|
| 50ep | 1.50% | 4.176 | 4.201 | 8.710 | 24.38 | 64.28 |
| 100ep | 2.93% | 3.357 | 4.569 | 8.475 | 25.20 | 63.65 |
| 150ep | 4.17% | 2.985 | 4.574 | 8.143 | 25.78 | 62.93 |
| 200ep | 6.38% | 2.841 | 4.523 | 7.758 | **25.88** | 62.72 |
| 300ep | 7.88% | 2.587 | 4.513 | 7.515 | 25.80 | 62.24 |

$$r(DL/\gamma_t,\ \mathrm{AA}) = \mathbf{-0.971},\qquad
r(DL/\gamma_t,\ \mathrm{clean}) = \mathbf{+0.977}.$$

**One quantity, derived from the inequality, tracks both axes** — and it has the sign that $O$ alone
gets wrong. Note also that $\mathbb{E}[\gamma_t]$ is essentially flat across the ladder ($4.20$ to
$4.57$): the ratio moves because $L$ falls, not because the teacher's margin grows. Longer teachers
are *easier to match under attack*.

⚠ **Two limitations, both real.**

*The certificate is vacuous.* It certifies $1.5$–$7.9\%$ against a measured AA of $24$–$26$, so it
is nowhere near tight and must never be presented as a bound that binds. Cauchy–Schwarz assumes the
displacement $\Phi_s(x')-\Phi_t(x)$ aligns with $w_y-w_c$, which is worst-case and far from what
happens; $\mathbb{E}[L]\approx7.8$ against $\mathbb{E}[\gamma_t]\approx4.5$ means the bound cannot
bite at all on the average sample. This is the second margin certificate in this project to come out
vacuous (T.6 retired one at $100.2\%$). **What survives is the ratio, not the guarantee.**

*It does not reproduce the saturation.* AA flattens after 150ep ($25.78/25.88/25.80$) while
$DL/\gamma_t$ keeps falling ($2.985/2.841/2.587$). The empirical ratio in `METHOD.md` §9.2,
rotation over class-mean gap, does flatten there ($1.254/1.259/1.258$) and correlates at $-0.991$.
So the derived quantity captures the trend and the hand-picked one captures the trend *and* the
plateau. Whether that is a defect of Cauchy–Schwarz slack or a sign that the right denominator is
the class-mean gap rather than the per-sample margin is open.

---

## T.3 The backbone target determines the representation

Theorem 1 says *which feature group* the anchored objective selects. This section says the objective
determines the representation **completely** — a property of $\mathcal{L}_{\mathrm{dir}}$ itself,
stated without reference to any baseline.

**The division of labor is measured before it is theorized.** Replacing the feature-space anchor by
the logit-space one — deleting $\mathcal{L}_{\mathrm{dir}}$ and letting the KD term reach the
backbone (`nofeat_champ200_norm`; the gradient routing changes, so this is a different anchored
method, not the champion minus one term) — leaves AA at $28.71$ vs $28.69$ and costs clean $-1.82$. Robustness belongs to the
anchor-plus-inner-maximization structure (T.1); what the **feature-direction form** of the anchor
buys is the clean axis, and this section is about how it earns it.

**Assumption T.3.** The backbone is linear, $\Phi_s(x)=Sx$, $S\in\mathbb{R}^{d\times m}$, and the
head does not influence it. Prop. 0 makes the second half **exact**: `featdir_alpha` $=0$ means the
head reads a detached feature, so the backbone is fit by $\mathcal{L}_{\mathrm{dir}}$ alone.

**Theorem 3 (identifiability).** *The backbone objective
$\mathbb{E}\lVert Sx_{\mathrm{adv}}-\Phi_t(x)\rVert^2$ is strictly convex in $S$ whenever
$\mathbb{E}[x_{\mathrm{adv}}x_{\mathrm{adv}}^\top]\succ0$, so its minimizer is unique: every one of
the $dm$ backbone parameters is determined by the target.*

*Proof.* The map $S\mapsto Sx$ has trivial kernel in $S$ once $\mathbb{E}[xx^\top]$ is full rank, so
the quadratic form is positive definite. $\square$

*Contrast.* A logit-space loss does not fix $S$: it fixes $WS$. Any $(S,W)\mapsto(AS,WA^{-1})$
leaves every logit unchanged, so the minimizer is determined only up to that reparametrization,
whereas an external feature target leaves no such freedom.

**Why this is the load-bearing property — and note what it is *not* about.** Theorem 3 is stated for
$\mathbb{E}\lVert Sx_{\mathrm{adv}}-\Phi_t(x)\rVert^2$, so it holds for the raw and the directional
target alike: it is an argument for anchoring in **feature space rather than logit space**, and it
says nothing about which feature-space target to use. That is deliberate — the design axis ties
(RESULTS.md §5e) and this section does not depend on it.

The content is uniqueness, and nothing more: a logit-space objective fixes $WS$, so it pins the
backbone only up to $(S,W)\mapsto(AS,WA^{-1})$, while an external feature target pins $S$ itself.
Since the head is re-solved on the student's own features (Theorem 2), the pair that a logit-space
objective settles on is one representative of that family, chosen by initialization and by the
trajectory rather than by the objective.

Robustness is set by Theorem 1 and is indifferent to the directions in question; clean accuracy is
not. **That asymmetry is the P2 signature — matched robustness, recovered natural accuracy —
predicted by the model and confirmed by measurement.**

*Remark (what the difference actually is).* The two targets differ in how much they specify per
sample: the feature target is $\Phi_t\in\mathbb{R}^{d}$, $d=512$ numbers, the logit target is
$z_t/\tau\in\mathbb{R}^{K}$, $K=100$. That is a difference in the amount of supervision, and
Theorem 3 says the richer one suffices to determine the backbone uniquely. It is **not** a claim
that one constrains feature directions the other leaves free: a logit-space loss lives in logit
space, and counting feature directions against it is a category error. Whether the richer target is
*better* is empirical, and our own measurement is modest — adding the feature term to the same
recipe buys clean $+1.82$ at an AA tie ($60.74/28.69$ vs `nofeat_champ200_norm` $58.92/28.71$).
⚠ *An earlier version of this remark argued a rank bound, that a $K\times d$ head leaves $m(d-K)$
directions unconstrained. That is wrong twice over — $W$ is trained too, so its row space is not
fixed, and anything the head cannot read is also invisible to the decision, so "unconstrained"
directions were never going to reach it. Deleted 2026-08-22.*

**Measurement (2026-08-08, full test set, champion-family runs only).** The observable for "how much
of the teacher's clean geometry the target retains" is $\cos(\hat\Phi_s(x),\hat\Phi_t(x))$. It orders
the runs exactly as clean accuracy does, while the adversarial geometry stays fixed:

| | $\cos(s,t)$ | clean | $\Delta=\lVert\hat\Phi_s(x_{\mathrm{adv}})-\hat\Phi_t(x)\rVert$ | align |
|---|---:|---:|---:|---:|
| champion | **0.8345** | **62.16** | 0.709 | 0.237 |
| $p{=}0$ | 0.8245 | 60.76 | 0.705 | 0.223 |
| `rawfeat` | 0.8178 | 59.78 | 0.708 | 0.212 |

Two structurally unrelated interventions — a per-sample **attack** allocation and a **target**
normalization — move the same scalar and move clean accuracy with it, while $\Delta$ and the
alignment are constant to within noise. *(Clean columns are re-evaluations at measurement time; the
headline table (§7.0 of METHOD.md) quotes $62.17$ / $60.74$.)*

**Extended to 17 cells (2026-08-21).** The same measurement now spans every cell of the stack
decomposition and both designs, clean $57.96$–$62.46$:
$r(\cos_{\text{clean}},\text{clean}) = +0.83$ within the directional cells and $+0.87$ within the
raw ones. The relation is broader than the three champion-family runs it was registered on, and it
does not depend on the design.

⚠ **It is a correlation, not a mechanism, and one reading of it is now excluded.** Those 17 cells
differ in WA / AWP / $\varepsilon_{\mathrm{tr}}$ / freeze_lr, and each of those moves alignment
**and** accuracy on its own, so the cross-cell relation is common-cause confounded. In particular
$r(\cos_{\text{clean}},\mathrm{AA}) = -0.90$ (raw) / $-0.57$ (direction) must **not** be read as
"alignment is paid for in robustness": the one controlled pair available — angeps $p$, a one-line
config diff — moves alignment $+0.0104$ / $+0.0123$ with AA $+0.04$ / $+0.46$, i.e. the opposite way.
See Prediction 3b. What this section may claim is that the target which retains more of the
teacher's clean geometry sits at higher clean accuracy; it may not claim that retaining it costs
robustness, nor that the alignment is what *causes* the clean gain.

**Reading — preservation is a *generalization* statement, not a fitting statement.** At
$x_{\mathrm{adv}}$, the point where $\mathcal{L}_{\mathrm{dir}}$ acts, the directional and raw
targets fit their own objectives equally well in angle ($\cos_{\mathrm{adv}}$ $0.7243$ vs $0.7242$
under a matched adversary). The gap exists **only on clean inputs**: the directional target does not
fit better — its alignment *generalizes to the clean point* better. Quantitatively the clean-side
gap carries the right order: $+0.0067$ in $\cos$ implies $\approx+0.8\%$ relative in the decision
statistic, against $+1.6\%$ relative measured clean ($59.79\to60.74$) — the same order of magnitude,
which is the most this measurement can claim. *(This is the surviving, clean-axis half of the
retired SNR account — see T.6.)*

---

## T.4 The magnitude coordinate — structural facts only

⚠ **Demoted 2026-08-21.** This section used to argue that the directional target is the better
design. It is not: at each design's own best schedule the two tie (NRR 39.17 vs 39.29, RESULTS.md
§5e), and in the bare regime the raw target wins on *both* axes (62.40 / AA 24.34 against 61.52 /
22.90). What survives is structural and does not depend on that comparison: **Prop. 4** (the head
reads the magnitude with weight exactly zero) and **the 2×2 algebra** (whether the objective is
magnitude-free is decided entirely by the student side; the teacher side only toggles a per-sample
weight $\lVert\Phi_t\rVert$), the latter confirmed three times on the grid. Read the rest of this
section as those two results plus the measurements that motivated them.

⚠ **Order of the argument — do not invert it.** Normalizing the head input
($z_s=W_s\hat\Phi_s+b_s$, `student_norm: True`) is **a design choice of this method**, not a
property of the architecture we may assume. Everything in this section is therefore *conditional on
having chosen the directional pipeline*; it explains an ablation **inside** the design and must not
be presented as the reason for adopting the design. The earlier draft made exactly that inversion,
which is circular.

**Why direction is chosen — the premise stated in its defensible (negative) form.** The tempting
premise is the positive one: *"class information lives in the direction, the magnitude is a
difficulty scalar."* **We do not claim that**, because this project already measured against it: the
norm-adaptivity experiments found the feature-norm dispersion on CIFAR small enough that a
norm-driven temperature is effectively constant, so the magnitude carries little of *either*
quantity. Asserting "the norm carries difficulty" invites refutation by our own data.

The defensible premise is the negative one, and it is all the argument needs:

> The magnitude is **not used by the decision** (given the design) and is **not needed for
> robustness** (Theorem 1 holds with no normalization anywhere in it).

**The positive premise that *does* survive — measured 2026-08-09, and the measurement rewrote it:**

1. **What the attack actually does to the feature** (PGD-CE 10 steps, $\varepsilon=8/255$,
   $n=1024$, `measure_attack_action.py`). The DGP inequality
   $\cos\angle(\Phi_c,\Phi_a)<\lVert\Phi_a\rVert/\lVert\Phi_c\rVert$ is **confirmed on the natural
   teacher** (99.9 % of samples) — but not as "rotation-dominance": the teacher's feature rotates
   $62°$ *and* its norm **inflates ×2.47** (the confidence coordinate is the most $\varepsilon$-volatile
   thing in the picture — the Ilyas confidence-rides-non-robust-features story, directly measured).
   On AT students the inequality holds for only ~41 % (12–13° rotation, ~5 % norm change, both
   small). Two retired readings, both dead: the 2026-07-22 invariance lemma ("direction is more
   invariant") — false, direction is less invariant on the teacher; and the registered
   damage-size prediction ("rotation exceeds rescaling", plan backlog 5) — **rejected as worded**,
   the norm moved more.
2. **Norm ancillarity, now with teeth.** In the Gaussian feature model
   $\mathbb{E}[\lVert\Phi\rVert^2\mid y]$ is $y$-free; on CIFAR the norm dispersion is too small to
   carry much of anything (norm-adaptivity result); and the measurement above adds: the natural
   feature's norm is simultaneously **class-empty and its most attack-volatile coordinate**
   (×2.47 under $\varepsilon$-perturbation, against 5 % on a robust net).

Together (post-measurement, weaker and honest): the class information rides the direction; the
magnitude of a natural feature is an information-free coordinate whose value is an artifact of the
non-robust bulk. Matching the direction targets the class-carrying coordinate; there is nothing on
the magnitude axis worth anchoring to. **This is stated as an observation supporting the design
choice, not as a mechanism** — why binding the empty coordinate *costs* clean remains the
adjudication below.

**The right form of the claim — constraint count, not fitting difficulty.** The $L_2$ target is the
cosine target **plus one additional constraint**, and the added constraint binds a coordinate the
decision reads with weight exactly zero (Prop. 4). Two tempting readings are measured false and must
not be written: *(i)* "the direction is easier to learn" — at $x_{\mathrm{adv}}$ both targets fit
the angle identically ($0.7243$ vs $0.7242$), so nothing was easier; *(ii)* "the doubled target
cannot be fit as well" — loss values are not comparable across objectives, and the angular
component, the only decision-visible one, shows no fitting gap. What the extra constraint changes is
not the achieved fit but the **selected solution**: among the many backbones that fit the
adversarial angle equally well, the norm-bound objective selects one whose alignment generalizes
worse to clean inputs (T.3: $\cos_{\text{clean}}$ $0.8178$ vs $0.8245$, clean $-0.95$).
Constraining a decision-null coordinate cannot help the decision; measured, it costs
generalization.

**The 2×2, expanded — the student side is the load-bearing one, by algebra.** With the teacher
detached, expand each cell of the normalization grid in $\theta=\angle(\Phi_s,\Phi_t)$:

| cell | loss | command to $\lVert\Phi_s\rVert$ |
|---|---|---|
| $\hat s,\hat t$ (champion) | $2-2\cos\theta$ | none — direction only |
| $\hat s,\Phi_t$ (`trawsnorm`) | $1+\lVert\Phi_t\rVert^2-2\lVert\Phi_t\rVert\cos\theta$ | none — a cosine loss with per-sample weight $\lVert\Phi_t\rVert$ |
| $\Phi_s,\hat t$ (`tnorm_sraw`) | $\lVert\Phi_s\rVert^2-2\lVert\Phi_s\rVert\cos\theta+1$ | $\lVert\Phi_s\rVert\to\cos\theta\le1$ |
| $\Phi_s,\Phi_t$ (`rawfeat`) | $\lVert\Phi_s-\Phi_t\rVert^2$ | $\lVert\Phi_s\rVert\to\lVert\Phi_t\rVert\cos\theta$ |

Whether the objective is magnitude-free is decided **entirely by the student side** (degree-0
homogeneity in $\Phi_s$ holds iff the student is normalized); the teacher side only toggles a
per-sample confidence weight $\lVert\Phi_t\rVert$. This answers "why must the *student* be
normalized when the teacher already supplies the direction?": a raw student side turns the same
unit-vector target into a magnitude command. Student normalization is not a trick on top of
direction matching — **it is what direction matching means.** The table's predictions against the
measured grid (champion regime): top row ties (measured $60.74$ vs $60.26$; the $-0.48$ is the
weighting effect), bottom row pays the magnitude cost (`rawfeat` $-0.95$ measured; `sraw_tnorm`
**predicted before any run** — clean down, $\lVert\Phi_s\rVert$ crushed toward $\cos\theta<1$).
The full 2×2 is being re-run at the **paper-base** regime with AA on every cell — predictions
registered in `theory_experiment_necessary.md`.

~~The supporting comparison is `featdir_champ200_fullraw` … worse on both axes. That, not Prop. 4,
is the argument for the design.~~ **Retracted 2026-08-21.** That $57.78$ / $28.04$ was the raw design
running on a recipe tuned for the directional one; `freeze_lr_epoch` alone costs it NRR $0.46$.
Given its own schedule the raw design reaches $62.35$ / AA $28.68$ / NRR $39.29$, which *dominates*
the champion. There is no design argument here to make. *(All comparisons in this section are against the $p{=}0$ champion
`featdir_champ200_100ep`; the angeps champion differs only in attack allocation, which T.5 owns.)*

**The trade-off argument, in one line.** Robustness is set by the $\ell_1$ suppression of Theorem 1,
whose derivation contains no normalization; clean accuracy is set by how well the adversarial
alignment **generalizes to clean inputs** (T.3). *The two axes are governed by different quantities,
so a gain on one carries no obligation to lose the other.* This is the whole of the trade-off claim,
and it is what the AA-tie-plus-clean-gain signature registers — `rawfeat` (clean $-0.95$, PGD
$34.94$ vs $34.94$), `nofeat` ($-1.82$ clean, AA $+0.02$), angeps ($+1.43$ clean, AA $-0.10$).
⚠ Attributing the clean half specifically to a *magnitude constraint* is what does not survive: the
alignment/robustness correlation that motivated it is confounded across regimes (Prediction 3b), and
the design axis itself ties.

**Stated as an operating-point claim.** What survives of this paragraph is the *first* half only:
every anchor-side intervention lands at an **AA tie** and moves only clean — `rawfeat` $-0.95$,
`nofeat` $-1.82$, both within $\pm 0.12$ AA of $28.69$. That pattern is real and is the shape of the
paper's claim: these are clean-axis levers, not robustness levers.

⚠ The *second* half — that the directional anchor therefore dominates the raw one on the trade-off
curve — is **retracted 2026-08-21**. It was measured with the raw design on the directional design's
schedule. Given its own, the raw design reaches NRR $39.29$ against the champion's $39.17$ and
dominates it on both axes. Half of even the champion-recipe gap was the head fit rather than the
representation (equalized head refit: clean $+2.66 \to +1.37$, AA $+0.42 \to +0.15$, RESULTS.md §5g).
Normalization carries **no** trade-off claim over the raw target; what it carries is
well-posedness — the raw design has failure modes ($\lVert\Phi_s\rVert$ collapse to $0.60$ under a
unit-norm target; a warm-started raw student facing a $z_t/16$ target starts $16\times$ off and
spends most of its budget recovering) that $\lVert\hat\Phi_s\rVert = 1$ makes impossible.

**Gradient geometry — what is derivable about *how* the directional objective trains.** From
$\nabla_{\Phi_s}\mathcal{L}_{\cos}=-\tfrac{2}{\lVert\Phi_s\rVert}(I-\hat\Phi_s\hat\Phi_s^\top)\hat\Phi_t$,
three statements, none of which assumes anything about the head (no circularity):

1. **Pure rotation (exact).** The projector annihilates the radial component: every unit of the
   $\cos$ gradient moves the angle. The $L_2$ gradient $2(\Phi_s-\Phi_t)$ splits into a tangential
   part $2\lVert\Phi_t\rVert\sin\theta$ and a radial part $2(\lVert\Phi_s\rVert-\lVert\Phi_t\rVert\cos\theta)$
   — and the radial part moves the angle by **exactly zero**, while the angle is the observable that
   tracks clean accuracy (T.3, a measurement, not a design assumption). $L_2$ spends a measurable
   fraction of its gradient budget on a direction inert to that observable.
2. **Bounded influence (exact).** Per sample, $\lVert\nabla\mathcal{L}_{\cos}\rVert\le2/\lVert\Phi_s\rVert$
   — bounded; $\lVert\nabla\mathcal{L}_{L2}\rVert=2\lVert\Phi_s-\Phi_t\rVert$ — unbounded. Direction
   matching is a bounded-influence (robust-statistics) estimator on the sphere: no single
   hard/heavily-attacked sample can hijack the batch gradient. Registered signature: heavy-tailed
   per-sample gradient norms under $L_2$, capped under $\cos$.
3. **Auto-tempering (weak here — flag it).** The $1/\lVert\Phi_s\rVert$ factor is a per-sample step
   size (confident samples rotate less), the BN-style effective-LR argument. Our own norm-dispersion
   measurement says this weight is nearly constant on CIFAR, so this is a footnote, not a claim.

**Status of all three:** characterizations of the optimization, not proofs of the clean gain. They
are the *candidate mechanisms* of the dynamics account: if the norm-penalty cell holds clean
(Phase 2, `theory_experiment_necessary.md`), the radial:tangential ratio (1) and the gradient-norm
tail (2), measured on the 2×2 checkpoints, decide which one carries the effect.

**Proposition 4 (consequence, not premise).** *Given the normalized head, $\lVert\Phi_s(x)\rVert$ has
exactly zero effect on the classifier output, and $\mathcal{L}_{\mathrm{dir}}$ exerts zero force on
it: $\langle\Phi_s,\nabla_{\Phi_s}\lVert\hat\Phi_s-\hat\Phi_t\rVert^2\rangle=0$.*

*Proof.* The loss depends on $\Phi_s$ only through $\hat\Phi_s$, hence is homogeneous of degree $0$;
Euler's identity with $k=0$ gives the claim. $\square$

**Assumption T.4 (candidate mechanism — capacity).** Under a fixed representational budget,
constraining a decision-irrelevant degree of freedom costs the decision-relevant solution. Stated as
a claim about the *fit*, this is already contradicted (the angular fit at $x_{\mathrm{adv}}$ is a
tie); if it survives, it survives as a claim about the selected solution's **generalization**. It is
one of the two open candidates below, not an established premise.

**Corollary 4 (the verified part is the split, not the mechanism).** *Within* the directional
design, an $\mathcal{L}_{L2}$ backbone target additionally constrains $\lVert\Phi_s\rVert$ — a null
direction of the decision — and by Theorem 1 (which is normalization-free) that constraint costs no
robustness. That it costs clean accuracy is **measured** ($-0.95$); *why* it does is the open item.

**Verified**: `rawfeat`, which changes the target only and leaves the head reading $\hat\Phi_s$ —
clean $-0.95$, PGD-20 $34.94$ vs $34.94$, AA $-0.12$. Prop. 4 is also directly visible:
$\lVert\Phi_s\rVert=34.4$ under $\mathcal{L}_{\cos}$ versus $8.7$ under $\mathcal{L}_{L2}$
(teacher $11.2$) — the raw target really does pull on a quantity the decision cannot read.

**Open item — it may be scale, not capacity.** Assumption T.4 attributes the clean cost to capacity
competition. The measurement points somewhere else: what the $L_2$ target binds is not the *spread*
of the norm across samples (small on CIFAR, per the norm-adaptivity result above) but its overall
**scale** — $34.4$ versus $8.7$, a factor of four. The direction gradient carries an overall factor
$\lVert\Phi_s\rVert^{-1}$ (visible in Prop. 4's proof), so the two runs fit the angle at effective step sizes differing by that
same factor. An optimization-dynamics account would then explain the $-0.95$ without invoking
capacity at all. Neither account is established; no third hypothesis is offered here. What is
measured is the split itself (clean moves, robustness does not), and that is what the paper claims.

**⚠ ADJUDICATION OUTCOME (2026-08-09): P-C — the diagonal inverts at paper-base.** *(Erratum
2026-08-10: the base cells vary the backbone target only — `student_norm: True` throughout, so
head input and inference stay normalized. The verdict below was established for that **partial**
raw. **Closed 2026-08-18**: the full raw design at base now exists — `wadec_raw_nowa`
$62.40$/AA $24.34$, within $0.05$ AA of the partial cell. The verdict is unchanged and now rests on
the consistent design.)* With the stack
removed (no WA/AWP/angeps, $\varepsilon_{\mathrm{tr}}=\varepsilon$, 100ep), the raw target beats
the directional one on **both axes**: $62.54$/AA $24.29$ vs $61.52$/AA $22.90$. The 50ep grid was
right, and it was not a PGD artifact. Consequently: **the directional clean advantage this section
explains is a property of the method-plus-stack, not of the loss in isolation** (with the stack,
direction wins clean $+0.95$ at an AA tie; without it, direction loses both). Every claim above
must be read at the champion regime — which is where the paper's numbers and all SOTA baselines
live — and the open mechanism question changes from "why does the norm constraint cost clean" to
**"why does the directional loss convert the stack better"** (stack lifts direction $+5.79$ AA vs
raw $+4.28$; which element carries the interaction is unmeasured, WA first suspect). The npen cell
is moot at base per the registered rule.

**Adjudication plan** (plan of record with registered predictions: `theory_experiment_necessary.md`).
⚠ **Regime caveat first**: every number above sits on the champion stack (WA, AWP,
$\varepsilon_{\mathrm{tr}}=8.8/255$, frozen-LR tail), and the 50ep *no-stack* grid **inverted** the
ordering (raw won PGD at tied clean). Whether the clean effect is design-intrinsic or a stack
interaction is therefore open, and is what Phase 1 settles. The champion-regime `npen01`/`tnorm_sraw`
runs were canceled (2026-08-09) in favour of:

1. **Paper-base 2×2** (no WA / no AWP / no angeps / $\varepsilon_{\mathrm{tr}}=\varepsilon$; AA on
   every cell; **running**): P-A — top row ties, bottom row pays clean at an AA tie → the effect is
   design-intrinsic and this section stands on the clean base. P-B — all four tie → the effect is a
   **stack interaction**; rewrite this section as a conditional claim. P-C — raw wins AA → major
   revision of T.3/T.4. Champion-regime data consistent so far: `trawsnorm`
   $60.26/35.33/30.49$/AA $28.63$ (clean $-0.48$, the weighting effect only).
2. **$\mathcal{L}_{\cos}$ + explicit norm penalty** $\mu(\lVert\Phi_s\rVert-\lVert\Phi_t\rVert)^2$
   at paper-base (`featdir_norm_penalty`, implemented) — **conditional on P-A**: clean drops →
   binding the norm *per se* is the cost (capacity); clean holds → the raw gradient's geometry is
   the cost (dynamics), then per-sample gradient statistics name it (radial leakage vs weighting).
3. `rawfeat` LR sweep — blunt under AdamW (per-parameter normalization absorbs global gradient
   scale); only if 2 is ambiguous.

---

## T.5 Sensitivity-matched $\varepsilon$ (allocation rule, not a theorem)

The objective is optimized over a **pixel** ball while what it measures is something else entirely,
and a fixed pixel radius moves that something by wildly different amounts across samples. The
allocation follows from one elementary fact: inside an $\ell_\infty$ ball of radius $e$ the
first-order change of any differentiable loss is

$$\max_{\lVert\delta\rVert_\infty\le e}\langle\nabla_xL,\delta\rangle=e\lVert\nabla_xL\rVert_1,$$

the dual norm of $\ell_\infty$ being $\ell_1$. Equalizing that movement subject to a fixed total
budget $\sum_i\varepsilon_i=N\varepsilon$ gives $\varepsilon_ig_i=\text{const}$, i.e.
$\varepsilon_i\propto1/g_i$ with $g_i=\lVert\nabla_xL(x_i)\rVert_1$ — equivalently the max–min
allocation, since any unequal assignment can be improved by moving budget to the sample that moves
least. $p=1$ is that allocation, $p=0$ is uniform-pixel, $p\in(0,1)$ interpolates. Under box
constraints $w\in[w_{\lo},w_{\hi}]$ the exact solution is the scale $t$ solving
$\sum_i\mathrm{clip}(t\,r_i,w_{\lo},w_{\hi})=N$, which reduces to mean-restoration when no clip binds.

**Budget preservation is the load-bearing detail.** $\sum_i\varepsilon_i=N\varepsilon$ exactly, so a
$p{=}1$ run and a $p{=}0$ run spend the same total attack budget and any difference between them is
*allocation*, not strength. Without it every gain is answerable by "you attacked less."

**Three corrections to how this was stated before (2026-08-22).**

1. **It equalizes the loss's movement, not an angle.** For $L_{\mathrm{dir}}=2-2\cos\theta$ the
   quantity held equal is $\Delta\cos\theta$; converting to $\Delta\theta$ carries a per-sample
   factor $1/\sin\theta$. Calling it an *angular* budget is therefore an approximation even in the
   directional design, and the honest name is **sensitivity-matched $\varepsilon$**.
2. **It is not specific to the directional objective.** $\lVert\nabla_xL\rVert_1$ is defined for any
   loss, and the implementation reads whatever loss is configured
   (`methods.py`: `_fsh = _fs if raw_s else normalize(_fs)`). We have run it on the raw-$L_2$
   objective, where it equalizes a *feature displacement* and works
   ($60.51$ / AA $28.59$ against $57.78$ / $28.04$ at $p{=}0$). The earlier claim that the rule "is
   only definable because the objective is directional" is false and is withdrawn.
3. **What actually separates it from IAAT / MMA / CAT** is therefore not directionality. Those
   assign a per-sample radius from *difficulty* or an *input-space margin*; this assigns it from the
   input-sensitivity of the training loss itself — the geometry the objective is written in rather
   than a property of the sample.

**Approximation to declare.** $g_i$ is measured once, by a single backward pass at the **clean**
$x_i$ before the attack starts (`methods.py`), so it is the sensitivity at the starting point, not
along the PGD trajectory. First-order in $\varepsilon$ throughout.

⚠ **Implementation note.** The 2026-08-04 champion used $\lVert\cdot\rVert_2$ where the derivation
asks for $\lVert\cdot\rVert_1$, and a clip-then-rescale that leaves the box (logged
$w_{\min}=0.40$ against $w_{\lo}=0.5$); both are switchable via `featdir_angeps_gnorm` /
`featdir_angeps_exact_budget`, defaulting to the old behaviour so every logged run reproduces.
**Resolved 2026-08-22**: the $\ell_1$ allocation was trained — $62.51$ / PGD $34.29$ / CW $30.63$ /
AA $28.44$ against the shipped $62.17$ / $34.77$ / $30.92$ / $28.59$, every difference inside the
noise band. The allocation is insensitive to the norm at first order, so the derivation may be
printed as stated with that measurement as a footnote.

---

## T.6 What has no theory here

Weight averaging, AWP, the one-cycle/frozen-LR schedule. They supply $+0.82$ / $+0.56$ AA and are
explained by nothing above; the paper's defence on that axis is the external comparison (ADR-full
carries the same machinery at twice the epoch budget). Stated plainly rather than papered over.

**This is now worse than "no theory" — it is where the design argument lives.** Prediction 5/5b: at
base the L2 target beats direction on both axes, and direction only wins once the stack is on, so
the stack is not a neutral booster sitting on top of the design — *it is the condition under which
the design wins at all*. WA is ruled out as that condition (2026-08-18: it pays both designs equally
and leaves L2 ahead by $+1.00$ AA). The remaining candidates are AWP, $\varepsilon_{\mathrm{tr}}=8.8/255$
and `freeze_lr_epoch`, which together give direction $+3.14$ AA against L2's $+1.49$. The live
hypothesis for AWP is **budget dilution**: AWP's weight perturbation has fixed norm
(`utils.py:159` rescales it to $\lVert w\rVert$), and under the L2 target part of the loss it
ascends lives on the magnitude coordinate — the one Prop. 4 shows the decision reads with weight
exactly zero — so a fixed-size adversarial weight perturbation spends part of itself where it
cannot change predictions.

**Narrowed to AWP (2026-08-18).** Adding one element at a time to the bare 100ep run, the AA gap
(positive = L2 ahead) moves $+1.44 \to +1.00$ (WA) $\to +0.70$ ($\varepsilon_{\mathrm{tr}}$ 8.8)
$\to -0.65$ (AWP $+$ freeze_lr). WA and the larger radius each tilt the same way but neither
reverses anything; the sign change and $1.35$ of the $2.09$ total swing are in the last step, as is
most of the champion regime's clean gap (that step pays direction $+0.78$ clean and costs L2
$-2.31$). This is what the dilution hypothesis predicts, and it is the **only** surviving candidate
mechanism for why the directional design wins at the operating point the paper reports.
**Resolved 2026-08-19.** Separating the last two: **AWP alone** takes the gap to a tie ($-0.07$,
direction clean $+1.20$); **`freeze_lr` alone** flips it outright ($+0.36$, clean $+1.46$); and the
two are **substitutes** — AWP's contribution to the gap falls from $0.63$ to $0.06$ once freeze_lr
is present. The dilution hypothesis is therefore at best half the story: AWP does pay direction more
($+1.77$ vs $+1.14$), but a frozen-LR tail pays it more still ($+1.84$ vs $+0.78$), and a
learning-rate schedule is not a mechanism this section can claim.

**What this costs the paper, stated plainly.** The design comparison has to be an operating-point
claim: in the bare regime the raw-$L_2$ target wins on *both* axes ($62.40$/AA $24.34$ against
$61.52$/$22.90$), so the sentence in T.4 calling `featdir_champ200_fullraw` "the argument for the
design" is not supportable as written and must be narrowed. What survives is stronger than that
sounds — the *signature*, AA tie with a clean gain, reproduces under AWP alone, under freeze_lr
alone, and under both: two independent elements, three recipes, and every regime a competitive AT
method actually publishes in (ADR itself carries WA $+$ AWP). What does **not** survive is the claim
that the advantage belongs to the loss rather than to the loss-plus-stack.

**What is design-intrinsic, and should carry that weight instead: well-posedness.** The raw design
has failure modes the directional one cannot have, because $\lVert\hat\Phi_s\rVert=1$ by
construction: norm collapse under a unit-norm target (`tnorm_sraw`, $\lVert\Phi_s\rVert\to0.60$,
AA $15.65$); first-epoch divergence when the head KD is the only term (`nofeat_raw`, clean $1.37$ at
step 0, from matching $\lVert\Phi_s\rVert\approx11$ logits against a $z_t/16$ target); and a
standing $\sim12\times$ head-KD sharpness mismatch in every full-raw cell. Those are properties of
the objective, not of the schedule, and they are already measured.

**The open mechanism question** is why every anti-overfitting element pays the directional design
more. The leading account is that it overfits more to begin with: in the bare regime $50\to100$ep
costs direction $1.86$ AA against $L_2$'s $0.67$, so a regularizer has more to recover — and
recovers past parity. Testable, not yet tested.

Also **retired and not to be reintroduced**: the margin certificate
$R_{\mathrm{rob}}\le\Pr[\hat\gamma\le\kappa]+\Pr[\Delta\ge\kappa]$ — measured vacuous at
$100.2\%$ for every $\kappa$ on all three runs (certified fraction $\approx6\%$ against PGD-20
$\approx35\%$); and the residual-variance/SNR account of normalization, whose key inequality
$\sigma^2_{\cos}<\sigma^2_{L2}$ is contradicted at $x_{\mathrm{adv}}$ by our own measurement. Only
its clean-input half survives — the measured $\cos_{\text{clean}}$ gap and its order-of-magnitude
match to the clean gain — and it appears in T.3 as a *reading of a measurement*, not as a derivation.

---

## Predictions summary

| # | statement | test | status |
|---|---|---|---|
| 1a | suppression of non-robust features has an $\varepsilon$ threshold | $\varepsilon_{\mathrm{tr}}$ sweep | partial (8.8 > 8 helps) |
| 1b | robust teacher changes nothing | robust-teacher ablation | **not run** (ckpts on disk) |
| 2 | head-freeze cost grows with $\varepsilon_{\mathrm{tr}}$ | freeze × $\varepsilon_{\mathrm{tr}}$, 3 points | **not run** |
| 3 | the target retaining more of the teacher's clean geometry gives higher clean at equal AA | $\cos_{\text{clean}}$ vs clean, now 17 cells | ✅ **correlation confirmed and broad**: $r=+0.83$ (direction) / $+0.87$ (L2), across designs and regimes. ⚠ **But not causal — see 3b.** |
| 4 | raw-magnitude target costs clean, not robustness | `rawfeat` | ✅ $-0.95$ clean, PGD tie |
| 3b | more clean-geometry alignment is *paid for* in robustness (the reading that made 3 a mechanism) | $\cos_{\text{clean}}$ vs AA | ❌ **not supported, 2026-08-21.** Across the 17 cells $r(\cos_{\text{clean}},\mathrm{AA})=-0.90$ (L2) / $-0.57$ (dir), but those cells differ in WA/AWP/$\varepsilon$/freeze_lr, and each of those moves alignment **and** AA on its own — textbook common-cause confounding. The **only** controlled pair (angeps $p$, a one-line config diff) goes the other way: alignment $+0.0104$/$+0.0123$ with AA $+0.04$/$+0.46$. No causal alignment-vs-robustness trade-off is established, so nothing needs a mechanism that "beats" one. |
| 5 | the clean cost is solution-selection, not fit — and which mechanism (and whether it is design-intrinsic at all) | paper-base 2×2 + npen (`theory_experiment_necessary.md`) | **P-C**: diagonal inverts at base (full raw $62.40$/AA $24.34$ > dir $61.52$/$22.90$, 2026-08-18) — the direction advantage is **stack-induced**; npen moot; new question = loss×stack interaction |
| 5b | *which* stack element induces it — WA was the registered suspect (direction gradient is pure rotation ⇒ trajectory aligned with weight averaging) | WA-only 2×2, C100 100ep (`wadec_*`) | ❌ **refuted 2026-08-18**: WA alone pays both designs nearly equally (+2.65 dir / +2.21 L2) and leaves L2 ahead **+1.00** AA (26.55 vs 25.55). The remaining three (AWP, $\varepsilon_{\mathrm{tr}}$ 8.8, freeze_lr) are worth +3.14 to direction vs +1.49 to L2 — the reversal lives there. $\varepsilon_{\mathrm{tr}}$ 8.8 also fails to reverse it (dir $+0.75$ / L2 $+0.45$, gap still L2 $+0.70$): **1.35 of the 2.09 gap swing, and the sign change, is AWP$+$freeze_lr**. Separated 2026-08-19: **AWP alone gives a tie** (dir clean $+1.20$ at AA $-0.07$) and **`freeze_lr` alone flips it** (clean $+1.46$ at AA $+0.36$); the two are substitutes, AWP's gap contribution falling $0.63\to0.06$ once freeze_lr is on. Not one fragile knob — but not design-intrinsic either. |

---

## T.7 Self-audit: what is worth printing, and what each claim is actually worth

Written 2026-08-22 after three claims in this file turned out to be wrong. Each entry states the
mathematical content, whether it is non-trivial, and the strongest objection a referee can raise.

### Worth printing

**(A) Theorem 1 — the adversarial term is an $\ell_1$ penalty on the non-robust coefficient.**
*Content:* the inner maximum is exactly $(\lvert R\rvert+\varepsilon\lvert b\rvert)^2$, so $J$ is
kinked at $b=0$ with subgradient jump $2\varepsilon\mathbb{E}\lvert R\rvert$; above a threshold
$\varepsilon_0$ the minimizer has $b^\star=0$ **exactly**, and by Cor. 2 it is then
$\varepsilon$-independent.
*Non-trivial:* yes, but narrowly. Exact sparsity is a kink phenomenon, not a generic consequence of
adding an adversary; and the result is **conditional** — below $\varepsilon_0$ the non-robust
feature is retained ($b^\star=0.518$ at $0.5\eta$), which makes it falsifiable.
*Objection to expect:* the model assumes $x_1$ unperturbable and $z$ fully perturbable, so *some*
suppression of $z$ is built in before anything is proved. **Concede this in the text.** What is not
built in is exactness, the threshold, and $\varepsilon$-independence.
*How to state it:* as an answer to the standing objection (a non-robust anchor imports non-robust
features) **inside the model where that objection was formulated** — never as "a natural teacher is
fine", which is existing practice and not a finding.

**(B) T.5 — sensitivity-matched $\varepsilon$ allocation.**
*Content:* inside an $\ell_\infty$ ball of radius $e$ the first-order movement of any loss is
$e\lVert\nabla_x L\rVert_1$, so $\varepsilon_i\propto(\bar g/g_i)^p$ equalizes it; the mean is
preserved, $\sum_i\varepsilon_i=N\varepsilon$, so the comparison against $p=0$ holds total attack
budget fixed.
*Non-trivial:* the derivation is elementary; the **idea** is not. The existing per-sample-$\varepsilon$
family (IAAT, MMA, CAT) allocates by *difficulty* or input-space margin. Allocating by the geometry
the training loss actually measures is the paper's one genuinely new mechanism.
*Objection to expect:* "you derive $\ell_1$ and ship $\ell_2$." Answered 2026-08-22: `gnorm 1`
gives $62.51/28.44$ against the shipped $62.17/28.59$, inside the noise band, so the derivation can
be printed with a footnote that the allocation is insensitive to the norm at first order.
*Budget preservation is the load-bearing detail* — without it every gain is answerable by "you just
attacked less."

### Printable but thin

**(C) Theorem 3 — uniqueness of the backbone under a feature-space target.**
*Content:* $S\mapsto\mathbb{E}\lVert Sx_{\mathrm{adv}}-\Phi_t\rVert^2$ is strictly convex when
$\mathbb{E}[x_{\mathrm{adv}}x_{\mathrm{adv}}^\top]\succ0$, hence a unique minimizer; a logit-space
objective fixes only $WS$ and therefore pins the backbone up to $(S,W)\mapsto(AS,WA^{-1})$.
*Non-trivial:* the convexity is elementary. The gauge contrast is correct and is the real content.
*Objection to expect — and we have no answer:* **why is uniqueness good?** Nothing here argues that
removing the reparametrization freedom improves anything; the empirical support is one comparison
(feature term added to the same recipe buys clean $+1.82$ at an AA tie) and that is correlational.
Assumption T.3 (linear backbone) is also strong.
*Do not restate this as a dimension count.* An earlier version claimed a rank-$K$ head leaves
$m(d-K)$ feature directions unconstrained. That is wrong: $W$ is trained, and anything the head
cannot read is invisible to the decision anyway. A logit loss lives in logit space; counting feature
directions against it is a category error.

**(D) Theorem 2 — the head cannot be inherited.**
*Content:* under Thm 1's solution the student's feature has gain $q$ and variance $q^2v$ against the
teacher's, so the teacher's Bayes head is not the student's; away from the scalar model the mismatch
ratio is $R=\sqrt{1-\rho^2}/\rho$, measured $0.77$ at $\varepsilon_{\mathrm{tr}}=8/255$ and $1.04$
at $8.8/255$.
*Non-trivial:* barely. "If the student's feature differs from the teacher's, the optimal head
differs" is close to definitional. The content is the *quantification* and Prediction 2 (the cost of
freezing the head grows monotonically in $\varepsilon_{\mathrm{tr}}$).
*Objection to expect:* that it proves the obvious. **This section needs its measurement or it should
be a remark, not a theorem** — `featdir_champ200_freezehead` and `featdir_alpha1_champion` are the
runs that decide which.

### Not printable

- **Any claim that the directional target beats the raw-$L_2$ one.** At each design's own schedule
  they tie (NRR $39.17$ vs $39.29$); in the bare regime raw wins both axes. Reproduced on a second
  machine. T.4 is demoted accordingly.
- **Any causal reading of $\cos_{\text{clean}}$ against robustness.** The cross-cell correlation is
  common-cause confounded and the one controlled pair points the other way (Prediction 3b).
- **Rank/dimension counting against a logit loss** (see C).
- **"A robust teacher is unnecessary."** It is a ranking claim against a line this paper does not
  compete with, and Cor. 1 does not support it. The claim is that a natural teacher is *usable*.
- **Anything from WA / AWP / the LR schedule.** T.6 owns this and owns that it has no theory —
  including the fact that `freeze_lr`, a schedule detail, is what used to make the design axis look
  decided.

### Honest summary

Two claims carry the section: **(A)** answers a standing objection in its own model, with a
falsifiable threshold, and **(B)** is a new allocation principle with the budget-preservation
argument that makes it non-vacuous. **(C)** is correct but its "so what" is unargued. **(D)** is
near-definitional until its measurement lands. Everything else in this file is either structural
bookkeeping or a record of something that did not survive.

---

## T.8 Referee pass (2026-08-22): the gaps, not the errors

T.7 audits whether each claim is *true*. This section asks the harder question — whether the section
would survive an ICML/ICLR referee. The errors are fixed; what follows are the holes.

### The central gap: Theorem 1 does not explain the result the paper sells

Theorem 1's solution is $b^\star=0$, $a^\star=q\eta$, i.e. $\Phi_s=q\eta\,x_1$ — the student uses the
robust feature and nothing else. Two consequences the section has never confronted:

1. **Clean accuracy in the model is capped at $x_1$'s, $95\%$,** against the teacher's $99.99\%$. The
   anchor does not recover any of the teacher's clean accuracy; it recovers none of it.
2. **Plain adversarial training reaches the same classifier.** Label-driven AT in this model also
   suppresses $z$ and predicts from $x_1$, and since the decision is $\mathrm{sign}(\Phi_s)$, the
   scale $a^\star$ is irrelevant to it. **Inside the model, anchoring changes the resulting
   classifier not at all.**

So Theorem 1 supports exactly one claim — *the anchor costs no robustness* — and is silent on the
claim the paper is actually selling, that the anchor **buys clean accuracy** ($57.36\to62.17$ against
ADR). A referee will put it plainly: the theory explains why the method does not fail, not why it
works.

**What a theory of the clean gain would need.** Room for the teacher's output to carry more than the
label does. The present model has a single robust feature, so "match the teacher's value" and "match
the label" select the same solution and there is nothing left for dark knowledge to be. A model with
several robust features of differing reliability — where a continuous target transmits the
reliability structure that a label cannot — is the smallest extension that could express the effect.
**Not attempted. This is the section's largest open item, and it is a modelling problem, not an
experiment.**

### T.3: the assumptions do the work

Strict convexity giving a unique minimizer is elementary; the content is the gauge contrast (a
logit-space objective fixes $WS$, not $S$). But Assumption T.3 — linear backbone, full-rank input
covariance — is what produces the convexity, and no deep network satisfies it. A referee reads that
as the conclusion being assumed. And the "so what" is still missing: **nothing anywhere argues that
removing the reparametrization freedom is beneficial.** The one empirical anchor (adding the feature
term buys clean $+1.82$ at an AA tie) is a single correlational comparison.

### T.5: the mathematics is one line and the baseline is missing

Two separate problems.

*The first-order argument is far from the operating regime.* $\max_{\lVert\delta\rVert_\infty\le e}\langle\nabla_xL,\delta\rangle=e\lVert\nabla_xL\rVert_1$
is textbook convex analysis, and it describes one infinitesimal step; the attack is 10-step PGD at
$\varepsilon=8.8/255$, and $g_i$ is measured once at the clean point before any of it. The rule is a
heuristic with a first-order motivation, and should be labelled as one.

*The comparison that defines the contribution has not been run.* T.5 differentiates itself from
IAAT / MMA / CAT by *what the radius is allocated from* — loss sensitivity rather than difficulty or
input-space margin. **No IAAT/MMA/CAT baseline exists in this project.** The novelty claim rests
entirely on a described difference to methods we have never measured against. At a venue of this
level that is not a weakness, it is a missing experiment, and it is the first thing to fix.

### Verdict

| | correct | non-trivial | supports the paper's claim |
|---|---|---|---|
| Thm 1 | ✅ (re-verified numerically) | conditional threshold, exact sparsity | ❌ only the negative half |
| Thm 3 | ✅ | elementary; assumptions carry it | ❌ "so what" unargued |
| T.5 rule | ✅ | the idea, not the derivation | ⚠ untested against its own comparison class |

As it stands the section contains one theorem about why the method does not fail, and two elementary
observations. That is not yet an ICML/ICLR theory section. The two things that would change it are
(i) a baseline against the per-sample-$\varepsilon$ family, and (ii) a model in which the anchor's
clean gain can exist at all.
