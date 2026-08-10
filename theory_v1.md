# Theory section (paper draft, 2026-08-08)

Formal statements only. Every result is followed by a **Prediction** line naming the experiment that
tests it; results with no testable consequence are not in this section. `method_v2.md` remains the
lab notebook — this file is what goes in the paper.

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

> **What Theorem 1 is for.** Guiding adversarial training with a clean-trained network to recover
> clean accuracy is established practice; the standing objection to it is that a *non-robust* anchor
> imports the teacher's non-robust features and therefore **taxes robustness** — the expectation
> under which ADR self-anchors and the ARD/RSLAD line requires a robust teacher. Theorem 1 says the
> tax is never collected: robustness is set by the inner maximization alone (the same source as in
> plain AT), and the anchor contributes only the value to be matched. **The anchor is a
> robustness-free knob** — which is what licenses spending it entirely on the clean axis (T.3–T.4).

**Numerically** (exact minimization of $J$, $d=512$, $n=4\cdot10^5$ Monte-Carlo):

| $\varepsilon$ | $b^\star$ | quadratic-relaxation bound |
|---|---:|---:|
| $0.5\eta$ | 0.518 | 0.543 |
| $1.0\eta$ | **0.000** | 0.229 |
| $2.0\eta$ (the model's radius) | **0.000** | 0.069 |
| $4.0\eta$ | **0.000** | 0.018 |

so $\varepsilon_0\in(0.5\eta,\,1.0\eta)$ and the shipped regime sits far above it. (Dropping the
$\lvert b\rvert$ kink gives the closed-form relaxation
$b^\star_{\mathrm{quad}}=c_1/(c_1+\varepsilon^2)$ with $c_1=\eta^2v/q^2+1/d$, a rigorous *upper*
bound; the exact answer is stronger.)

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
compromise *within this setting*. It must **not** be stated as "a robust teacher is unnecessary" —
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

**Theorem 2.** *Under Theorem 1's solution the student's feature is distributed as*

| | mean | variance |
|---|---|---|
| teacher $\Phi_t$ | $\eta y$ | $1/d$ |
| student $a^\star x_1$ | $q\eta y=0.9\,\eta y$ | $q^2v/1=\mathbf{3.04}/d$ |

*so the Bayes-optimal linear head for the teacher's feature is not optimal for the student's; the
gain mismatch is $q^{-1}$ and the noise mismatch is $3.04\times$.*

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

## T.3 The backbone target determines the representation

Theorem 1 says *which feature group* the anchored objective selects. This section says the objective
determines the representation **completely** — a property of $\mathcal{L}_{\mathrm{dir}}$ itself,
stated without reference to any baseline.

**The division of labor is measured before it is theorized.** Replacing the feature-space anchor by
the logit-space one — deleting $\mathcal{L}_{\mathrm{dir}}$ and letting the KD term reach the
backbone (`nofeat_champ200_norm`; the gradient routing changes, so this is a different anchored
method, not the champion minus one term) — leaves AA at $28.71$ vs $28.69$ and costs clean $-1.82$;
the raw variant of the same ablation diverges in the first epoch. Robustness belongs to the
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

*Contrast.* A loss that reaches the backbone only through a rank-$r$ map $g$ is invariant along
$\{\Delta:g\Delta=0\}$, of dimension $m(d-r)$ — it determines nothing there.

**Why this is the load-bearing property.** The head is *re-solved* on the student's own features
(Theorem 2), so the final classifier reads all $d$ feature directions. Any objective that constrains
fewer than $d$ of them leaves the remainder to initialization and to drift under the inner
maximization — and those directions still reach the decision once the head is refit. The champion's
target constrains all of them, and pins them to the teacher's clean solution.

Robustness is set by Theorem 1 and is indifferent to the directions in question; clean accuracy is
not. **That asymmetry is the P2 signature — matched robustness, recovered natural accuracy —
predicted by the model and confirmed by measurement.**

*Remark (contrast, not evidence).* A loss reaching the backbone only through a $K\times d$ head has
$\mathrm{rank}\le K$ and therefore leaves $m(d-K)$ directions unconstrained; on CIFAR-100,
$K/d=100/512$. We state this as the structural difference from label-driven adversarial training and
rest the empirical case on the published baselines, not on an internal re-implementation.

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
headline table (§7.0 of METHOD.md) quotes $62.17$ / $60.74$.)* ⚠ *All three runs are
champion-regime; at paper-base the direction/raw ordering inverts (T.4 adjudication outcome), so
this correlation is established only under the stack until the base checkpoints are measured.*

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

## T.4 The magnitude, *given* the directional design

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

The supporting comparison is `featdir_champ200_fullraw`, which reverts *both* the head input and the
target to raw features, i.e. abandons the directional design entirely: clean $57.78$ / AA $28.04$
against the $p{=}0$ champion's $60.74$ / $28.69$ — **worse on both axes**. That, not Prop. 4, is the
argument for the design. *(All comparisons in this section are against the $p{=}0$ champion
`featdir_champ200_100ep`; the angeps champion differs only in attack allocation, which T.5 owns.)*

**The trade-off argument, in one line.** Robustness is set by the $\ell_1$ suppression of Theorem 1,
whose derivation contains no normalization; clean accuracy is set by how well the adversarial
angular alignment **generalizes to clean inputs** (T.3), which is where the magnitude constraint
exacts its cost. *The two axes are governed by different quantities, so a gain
on one carries no obligation to lose the other.* This is the whole of the trade-off claim, and the
`rawfeat` split (clean $-0.95$, PGD $34.94$ vs $34.94$) is exactly its signature.

**Stated as an operating-point claim.** Every anchor-side intervention lands at an AA tie and moves
only clean: `rawfeat` $-0.95$, `nofeat` $-1.82$ (both within $\pm0.12$ AA of $28.69$); `fullraw`,
which abandons the design entirely, loses on both axes. At matched robustness the directional anchor
**dominates** the raw one on the clean/robust trade-off curve, and NRR is the scalar that registers
the domination. This — not a robustness mechanism — is the claim the normalization carries.

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
head input and inference stay normalized. The verdict below is established for that **partial**
raw; the full raw design at base is unmeasured. See `theory_experiment_necessary.md`.)* With the stack
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

## T.5 The angular budget (design rationale, not a theorem)

Equation (1) maximizes a **feature angle** while the constraint set is a **pixel** ball. To first
order the angle moved inside an $\ell_\infty$ ball of radius $\varepsilon$ is
$\varepsilon\lVert\nabla_xL_{\mathrm{dir}}\rVert_1$ (the dual norm of $\ell_\infty$ is $\ell_1$), so
equalizing the *angular* budget under a fixed total pixel budget $\sum_i\varepsilon_i=N\varepsilon$
is the max–min allocation $\varepsilon_i\propto1/g_i$, $g_i=\lVert\nabla_xL_{\mathrm{dir}}(x_i)\rVert_1$.
$p=1$ is that allocation, $p=0$ is uniform-pixel, $p\in(0,1)$ interpolates. With box constraints
$w\in[w_{\lo},w_{\hi}]$ the exact solution is the scale $t$ solving
$\sum_i\mathrm{clip}(t\,r_i,w_{\lo},w_{\hi})=N$, which reduces to mean-restoration when no clip is
active.

No theorem is claimed. What the paragraph buys is that the allocation is *the solution of a stated
problem* rather than a heuristic — and that it is only definable because the objective is
directional, which is what separates it from IAAT / MMA / CAT (difficulty- or margin-based radii).

⚠ **Implementation note.** The 2026-08-04 champion used $\lVert\cdot\rVert_2$ and a
clip-then-rescale that leaves the box (logged $w_{\min}=0.40$ against $w_{\lo}=0.5$). Both are fixed
behind `featdir_angeps_gnorm` / `featdir_angeps_exact_budget`, defaulting to the old behaviour so
every logged run reproduces. **The corrected allocation has not yet been trained**; until it is, the
paper must describe the shipped run as $\varepsilon_i\propto\lVert\nabla\rVert_2^{-1}$ and not claim
angular equalization.

---

## T.6 What has no theory here

Weight averaging, AWP, the one-cycle/frozen-LR schedule. They supply $+0.82$ / $+0.56$ AA and are
explained by nothing above; the paper's defence on that axis is the external comparison (ADR-full
carries the same machinery at twice the epoch budget). Stated plainly rather than papered over.

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
| 3 | the target retaining more of the teacher's clean geometry gives higher clean at equal AA | $\cos(s,t)$ vs clean, champion / $p{=}0$ / `rawfeat` | ✅ monotone; $\Delta$, align flat |
| 4 | raw-magnitude target costs clean, not robustness | `rawfeat` | ✅ $-0.95$ clean, PGD tie |
| 5 | the clean cost is solution-selection, not fit — and which mechanism (and whether it is design-intrinsic at all) | paper-base 2×2 + npen (`theory_experiment_necessary.md`) | **P-C**: diagonal inverts at base (raw $62.54$/AA $24.29$ > dir $61.52$/$22.90$) — the direction advantage is **stack-induced**; npen moot; new question = loss×stack interaction |
