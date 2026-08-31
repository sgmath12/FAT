# Method: Directional Feature Distillation from a Natural Teacher

*Working draft, 2026-08-01. Notation fixed here is meant to carry into the paper.*

---

## Positioning

Two commitments, both deliberate, both load-bearing:

**(P1) The teacher is naturally trained — and this is a claim, not a fallback.**
Robust knowledge distillation (ARD, RSLAD, ARREST, B-MTARD, …) requires an adversarially-trained
teacher, by far the most expensive precondition in that literature. We require only an ordinary
natural checkpoint. The argument of §2 and §4 is what licenses this: a natural teacher is precisely
a $(\mathcal{F}_t,w_t)$ pair whose feature set was selected by $\rho$ alone, and the method is built
to keep $\mathcal{F}_t$ (which is what supplies clean accuracy) while structurally discarding
$w_t$ and the unreachable non-robust component (which is what the adversarial term supplies
$\gamma$ against). $\rho$ comes from the teacher, $\gamma$ comes from the inner maximization, and
the two sources are separated by construction — so **the method does not depend on the teacher being
robust**. *(Scope: this is a mechanism statement at matched teacher capacity, not a ranking claim
against robust-teacher KD — that line buys its numbers substantially from* stronger *teachers
(large robust WRNs), an axis we do not compete on. Our comparison class is methods without a robust
teacher — ADR, TRADES-family — and §7 reports it as such.)*

**(P2) The target is the clean/robust operating point, not AA alone.**
The method buys $+2.7$ clean at a cost of $-1.1$ AA against the strongest baseline (§7), a corner
of the trade-off curve that robustness-only evaluation does not reward and that a practitioner
often prefers. All comparisons therefore report clean accuracy alongside every robust metric, and
NRR is the headline scalar.

The one-line thesis is *not* "make maximal use of what the clean teacher learned." It is the
sharper and partly opposite statement:

> A naturally-trained teacher's knowledge **cannot** be transferred as a whole. It splits into a
> transferable part (the feature set $\mathcal{F}_t$, carrying $\rho$) and a part that must be
> discarded — the combination $w_t$, which is $\gamma$-blind by construction, and the non-robust
> component $\Phi_t^{(n)}$, which is unreachable at $x_{\mathrm{adv}}$ by definition. **The method
> is the mechanism that enforces that boundary.** Every design choice below (feature space, direction
> only, stop-gradient, free head) is a corollary of it.

---

## 0. Notation

| symbol | meaning |
|---|---|
| $(x,y)\sim\mathcal{D}$ | image $x\in[0,1]^{n}$, label $y\in[K]$, $K=100$ (CIFAR-100) |
| $\mathcal{B}(x,\varepsilon)$ | $\{x' : \lVert x'-x\rVert_\infty\le\varepsilon\}\cap[0,1]^n$ |
| $\Phi_t:\mathcal{X}\to\mathbb{R}^d$ | **teacher** backbone (penultimate features), $d=512$ |
| $W_t\in\mathbb{R}^{K\times d},\ b_t$ | teacher classifier head; $z_t(x)=W_t\Phi_t(x)+b_t$ |
| $\Phi_s, W_s, b_s$ | student counterparts, parameters $\theta=(\theta_{\mathrm{bb}},\theta_{\mathrm{hd}})$ |
| $\hat v$ | $v/\lVert v\rVert_2$ (unit direction) |
| $Q\in\mathbb{R}^{d\times k}$ | fixed orthonormal matrix, $Q^\top Q=I_k$ |
| $\bar\theta$ | weight-averaged (EMA) shadow parameters — **the evaluated model** |
| $\varepsilon,\ \varepsilon_{\mathrm{tr}}$ | evaluation / training attack radius (they differ) |

The teacher is trained **naturally** (no adversarial training) and then frozen.
The student head consumes a *normalized* feature:

$$z_s(x)\;=\;W_s\big(\sigma\,\hat\Phi_s(x)\big)+b_s,\qquad \sigma=1 .$$

---

## 1. Starting point: the TRADES decomposition

Zhang et al. decompose the robust error into a natural term and a boundary term,

$$R_{\mathrm{rob}}(f)\;=\;R_{\mathrm{nat}}(f)\;+\;R_{\mathrm{bdy}}(f),$$

$$R_{\mathrm{nat}}(f)=\Pr[f(X)Y\le 0],\qquad
R_{\mathrm{bdy}}(f)=\Pr\big[X\in\mathcal{B}(\mathrm{DB}(f),\varepsilon),\ f(X)Y>0\big],$$

and upper-bound it (Thm. 3.1) by a differentiable surrogate, giving the TRADES objective

$$\min_{f}\ \underbrace{\mathbb{E}\,\mathrm{CE}\big(f(x),y\big)}_{\text{natural}}
\;+\;\beta\cdot\underbrace{\mathbb{E}\max_{x'\in\mathcal{B}(x,\varepsilon)}\mathrm{KL}\big(f(x)\,\Vert\,f(x')\big)}_{\text{boundary}} .
\tag{TRADES}$$

Two structural facts about (TRADES) drive everything below:

- **(T1)** both terms are functions of the *same* network $f$, and both gradients reach *all* parameters;
- **(T2)** the boundary term's anchor is the model's **own** clean output $f(x)$ — a *self-anchor* that drifts during training.

---

## 2. What the two terms are actually about (Ilyas et al.)

Ilyas et al. write a classifier as a **feature set plus a combination**, $C=(\mathcal{F},w,b)$, and grade features on two independent scales:

$$\textbf{$\rho$-useful:}\quad \mathbb{E}_{(x,y)}\big[y\cdot f(x)\big]\ \ge\ \rho,
\qquad
\textbf{$\gamma$-robustly useful:}\quad \mathbb{E}_{(x,y)}\Big[\inf_{\delta\in\Delta}y\cdot f(x+\delta)\Big]\ \ge\ \gamma .$$

A *useful non-robust* feature has $\rho>0$ but no $\gamma\ge0$. Their reading of adversarial training is explicit:

> minimizing the adversarial loss can be viewed as explicitly preventing the classifier from learning a **useful but non-robust combination of features**.

So AT does not delete features; **it re-solves $w$ under the $\gamma$ constraint.** Mapping onto our networks:

$$\mathcal{F}\ \longleftrightarrow\ \Phi_s\ (\theta_{\mathrm{bb}}),\qquad
w\ \longleftrightarrow\ W_s\ (\theta_{\mathrm{hd}}),\quad\text{row } w_c\in\mathbb{R}^d \text{ weights the } d \text{ features for class } c.$$

This yields the reading we build on:

- $R_{\mathrm{bdy}}$ asks *"is the representation stable under $\delta$"* → a statement about $\mathcal{F}$ → **backbone**.
- $R_{\mathrm{nat}}$ asks *"is the weighted vote correct on clean data"* → a statement about $w$ → **head**.

**(T1) is therefore a design choice, not a necessity.** Our method takes the opposite choice.

---

## 2b. Three claims, computed exactly (Tsipras et al. model)

Ilyas §2 gives vocabulary but no numbers. The Tsipras et al. Gaussian model gives both, and both of
our core claims fall out of it in closed form. Their data distribution:

$$y\sim\mathrm{Unif}\{\pm1\},\qquad
x_1=\begin{cases}y & \text{w.p. } p=0.95\\ -y & \text{else}\end{cases},\qquad
x_2,\dots,x_{d+1}\stackrel{iid}{\sim}\mathcal{N}(\eta y,1),\ \ \eta=\tfrac{4}{\sqrt d},$$

with an $\ell_\infty$ adversary of radius $\varepsilon=2\eta$. $x_1$ is the robust feature (bounded
$\varepsilon$ cannot flip a $\pm1$ coordinate); $x_2\dots x_{d+1}$ are the useful-but-non-robust bulk.

A **naturally trained teacher** uses the bulk:
$$\Phi_t(x)=\mathrm{mean}(x_2..x_{d+1}),\qquad \Phi_t\mid y\sim\mathcal{N}(\eta y,\tfrac1d),$$
giving standard accuracy $\Phi(\eta\sqrt d)=\Phi(4)\approx99.99\%$ and robust accuracy $\approx0$:
under attack every bulk coordinate shifts by $-2\eta y$, so $\Phi_t(x_{\mathrm{adv}})\sim\mathcal{N}(-\eta y,\tfrac1d)$ — the sign flips.

### Claim 1 — matching the teacher's **raw** value at $x_{\mathrm{adv}}$ selects the robust route

Our backbone objective in this model is $\min_s\max_{\delta}\big(\Phi_s(x+\delta)-\Phi_t(x)\big)^2$. Two candidate students:

| student | $\Phi_s$ | loss |
|---|---|---:|
| imitate the teacher (bulk mean) | $\mathrm{mean}(\text{bulk})$ | $(2\eta)^2+\tfrac2d=\tfrac{66}{d}$ |
| **robust coordinate only** | $\eta\,x_1$ | $\eta^2\cdot4(1-p)+\tfrac1d=\tfrac{4.2}{d}$ |

The robust route wins by $\approx16\times$. The mechanism: $\Phi_t(x)\sim\mathcal{N}(\eta y,1/d)$ is
**a proxy for $y$**. What is non-robust about the teacher is the *route* it takes to that value, not
the value. Demanding the same value at $x_{\mathrm{adv}}$ makes the min–max swap the route.

> This is why a **natural teacher suffices** (P1), stated as a computation rather than an intuition.

**Note what does not appear in the derivation: any normalization.** The loss is a squared difference
of raw values. Claim 1 therefore *predicts* that removing both normalizations changes nothing about
robustness — and that is what `featdir_champ200_rawfeat` measured: PGD-20 identical to three
significant figures (34.94 vs 34.94), AA $-0.12$, CW $-0.40$; only clean moved ($-0.95$). Feature
normalization is a **clean-accuracy lever, not a robustness mechanism.** The model is scalar so it
cannot explain the $+0.95$ clean; that gap is stated, not resolved.

### Claim 2 — the head, by contrast, **must** be re-solved

The two feature distributions are not the same:

| | mean | variance |
|---|---|---|
| teacher $\Phi_t(x)$ | $\eta y$ | $1/d$ |
| student $\eta x_1$ | $\mathbf{0.9}\,\eta y$ | $\mathbf{3.04}/d$ |

using $\mathbb{E}[x_1y]=2p-1=0.9$ and $\mathrm{Var}(x_1\mid y)=1-(2p-1)^2=0.19$. The student's
feature is shrunk $0.9\times$ in mean and **three times noisier**. A head calibrated for
$\mathcal{N}(\eta y,1/d)$ is not optimal for $\mathcal{N}(0.9\eta y,3.04/d)$.

So the two claims point in opposite directions, and that asymmetry *is* the method:

| | reuse the teacher's? |
|---|---|
| **feature** (backbone target) | **yes** — raw value, normalization optional |
| **head** | **no** — the student's feature distribution is different |

### Claim 3 (corollary) — *how* the head is re-solved is secondary

Claim 2 requires re-fitting on the student's own feature distribution; it does not privilege a
supervision signal. Hard CE and teacher KD both re-fit. Measured CW agrees:

| backbone | CE | KD | Δ |
|---|---:|---:|---:|
| raw | 27.09 | 27.24 | +0.15 |
| hat | 26.23 | 26.63 | +0.40 |

The $+3.1$–$3.5$ PGD-20 gap between the same pairs is not corroborated by CW, and the backbone in
each pair is **bit-identical** (`dir_loss_adv` 56.9736 in both `plain_tr_sr_{ce,kl}`; the detach
means the head cannot reach the backbone). A soft target that weakens CE-gradient attacks while
leaving margin attacks untouched is the signature of an artifact, not of robustness.

### The angular form of Claim 2, and its quantitative prediction

Away from the scalar model, write the student's adversarial feature direction in the teacher's frame.
For $\rho=\cos\angle(\Phi_s(x_{\mathrm{adv}}),\Phi_t(x))$, exactly:

$$\hat v=\rho\,\hat u+\sqrt{1-\rho^2}\;e,\qquad e\perp\hat u,\ \lVert e\rVert=1,$$
$$\langle w_c,\hat v\rangle=\underbrace{\rho\,\langle w_c,\hat u\rangle}_{\text{signal, shrunk}}
+\underbrace{\sqrt{1-\rho^2}\,\langle w_c,e\rangle}_{\text{contamination the teacher never saw}} .$$

$w_t$ is the $\rho=1$ solution. The mismatch ratio $R=\sqrt{1-\rho^2}/\rho$ measured on our runs:

| | $\rho$ | angle | $R$ |
|---|---:|---:|---:|
| 50ep grid, $\varepsilon_{\mathrm{tr}}=8/255$ | 0.793 | 37.6° | 0.77 |
| **champion, $\varepsilon_{\mathrm{tr}}=8.8/255$** | **0.693** | **46.2°** | **1.04** |

At the champion's operating point the contamination term is **larger than the signal**. And $e$ is
not random — it is adversarially chosen — so $\langle w_c,e\rangle$ cannot be dismissed as $O(\lVert w_c\rVert/\sqrt d)$.

**Prediction (sharper than "freezing hurts"): the cost of freezing the head should grow monotonically
as $\varepsilon_{\mathrm{tr}}$ rises**, because $\rho$ falls and $R$ grows. That is a curve, not a sign test.

### Caveats

1. $\Phi$ is a scalar in the model, 512-dimensional for us; the multi-dimensional version needs
   $\mathbb{E}[ee^\top]$ (the residual covariance) rather than a single $e$.
2. $x_1$ is assumed unperturbable. Real images have no such clean separation.
3. The model's teacher *is* the bulk mean. A real natural network mixes robust and non-robust
   features, which shrinks Claim 1's $16\times$ margin by an unknown amount.
4. $\rho$ drifts during training; "the optimal head at fixed $\rho$" is an approximation.
5. In the model, $p=0.95$ caps robust accuracy — i.e. the ceiling is set by the **data**, not by the
   teacher. That is a usable implication rather than a limitation.

---

## 3. The method

### 3.1 Initialization (the "finetune" step)

Both networks start from the same naturally-trained checkpoint $\theta_t$:

$$\theta_s^{(0)}\;=\;\theta_t\quad\text{(backbone \emph{and} head)},\qquad \theta_t\ \text{frozen thereafter}.$$

This is not a warm start of convenience. By §2 the natural checkpoint is exactly a $(\mathcal{F}_t,w_t)$ pair whose $\mathcal{F}_t$ was selected by $\rho$ alone; the student begins holding both, and the training procedure below is designed to *keep* $\mathcal{F}_t$ and *discard* $w_t$.

Note the two networks are not the same function at $t=0$: the teacher head sees $\Phi_t$, the student head sees $\sigma\hat\Phi_s$, so $z_s^{(0)}=z_t/\lVert\Phi_t\rVert$ up to the bias.

### 3.2 Inner maximization (attack)

$$x_{\mathrm{adv}}\;=\;\arg\max_{x'\in\mathcal{B}(x,\varepsilon_{\mathrm{tr}})}\ \big\lVert Q^\top\!\big(\hat\Phi_s(x')-\hat\Phi_t(x)\big)\big\rVert_2^2
\tag{1}$$

solved by $m$-step PGD with sign steps of size $\alpha$:

$$x^{(j+1)}=\Pi_{\mathcal{B}(x,\varepsilon_{\mathrm{tr}})}\Big(x^{(j)}+\alpha\,\mathrm{sign}\,\nabla_{x}\big\lVert Q^\top(\hat\Phi_s(x^{(j)})-\hat\Phi_t(x))\big\rVert^2\Big),\quad x^{(0)}=x+0.001\,\xi,\ \xi\sim\mathcal{N}(0,I).$$

Champion: $m=10$, $\alpha=2/255$, $\varepsilon_{\mathrm{tr}}=8.8/255$, $\varepsilon=8/255$ at evaluation.
**No label and no head — either network's — appears in (1).**

### 3.2a Sensitivity-matched $\varepsilon$ (`featdir_angeps_p`, 2026-08-04)

⚠ **Renamed 2026-08-25; the knob keeps its old name for log reproducibility.** This was called
"angular budget allocation", which is not what the rule does. What it equalizes is the first-order
movement of the **loss**; for $L_{\mathrm{dir}}=2-2\cos\theta$ that is $\Delta\cos\theta$, and
converting to $\Delta\theta$ carries a per-sample factor $1/\sin\theta$, so the angle is equalized
only approximately even in the directional design. Two further reasons the old name misleads: the
rule is not specific to a directional objective — it reads whatever loss is configured and has been
run on the raw-$L_2$ one, where it equalizes a feature *displacement* — and $g_i$ is measured by one
backward pass at the **clean** $x_i$, so it is the sensitivity at the starting point of a multi-step
PGD rather than along it.

**2026-08-31 update: the old name is now wrong twice over.** The shipped design is the unnormalized
$\ell_2$ anchor, which contains no angle at all — $\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_t(x)\rVert^2$
has nowhere for a $\theta$ to appear. The first objection above applied even to the directional
design; this one applies only now.

The mismatch it addresses is real regardless of the name: the constraint set
$\mathcal{B}(x,\varepsilon_{\mathrm{tr}})$ is a **pixel** ball while the loss measures something
else, and the same radius moves that something by wildly different amounts across samples. So we
equalize the loss's movement rather than the pixel budget.

Inside an $\ell_\infty$ ball of radius $\varepsilon$ the first-order change of any loss is
$\max_{\lVert\delta\rVert_\infty\le\varepsilon}\langle\nabla_xL,\delta\rangle=\varepsilon\lVert\nabla_xL\rVert_1$
(the dual norm of $\ell_\infty$ being $\ell_1$), so with
$g_i=\lVert\nabla_x L_{\mathrm{dir}}(x_i)\rVert_2$ and $\bar g$ the batch mean,

$$\tilde w_i=\mathrm{clip}\Big(\big(\bar g/g_i\big)^{p},\,w_{\lo},\,w_{\hi}\Big),\qquad
w_i=\tilde w_i\cdot\frac{N}{\sum_j \tilde w_j},\qquad
\varepsilon_i=\varepsilon_{\mathrm{tr}}\,w_i,\quad \alpha_i=\alpha\,w_i .$$

$p=0$ recovers the uniform champion; $p=1$ is full equalization. Three properties matter:

1. **The mean is restored *after* clipping**, so $\sum_i\varepsilon_i=N\varepsilon_{\mathrm{tr}}$
   *exactly*. This run and the $p=0$ champion spend the **identical total budget** — any difference
   is **allocation**, not a stronger or weaker attack. This is what structurally forecloses the
   "you just attacked more/less" reading, and it is the reason the comparison is clean.
2. $\alpha$ scales by the same $w_i$, holding **steps-per-radius** fixed, so the inner problem is
   solved to the same relative precision for every sample.
3. $g_i$ costs **one forward/backward on clean $x$** before the attack — $\sim$10 % of a 10-step
   PGD, and it uses the same $Q$-projected objective as (1), not a surrogate.

`inner_featdir_only_return` broadcasts, so $\varepsilon,\alpha$ enter as $[N,1,1,1]$ tensors through
the unchanged scalar code path.

**Relation to prior per-sample-$\varepsilon$ work.** Per-sample radii are an established family —
IAAT ([1910.08051](https://arxiv.org/abs/1910.08051)), MMA
([1812.02637](https://arxiv.org/abs/1812.02637)), CAT
([2002.06789](https://arxiv.org/abs/2002.06789)) — but all of them set the radius from
**difficulty or input-space margin**. Ours is set from the **geometry the loss lives in**, and is
only definable *because* the objective is directional. It is not a generic AT trick ported in; it
is a consequence of §3.2.

Champion: $p=1$, $w_{\lo}=0.5$, $w_{\hi}=1.5$. The clip is live — measured $w$ occupies
$[0.40,1.29]$ with $\mathrm{CV}(g)\approx0.65$–$0.74$ on CIFAR-100 and $0.84$–$0.86$ on CIFAR-10 —
so **the clip range is an effective hyper-parameter**, and widening it is untested.

### 3.3 Outer objective

$$\boxed{\ \mathcal{L}\;=\;\underbrace{\mathbb{E}\big\lVert Q^\top(\hat\Phi_s(x_{\mathrm{adv}})-\hat\Phi_t(x))\big\rVert^2}_{\mathcal{L}_{\mathrm{dir}}\ \to\ \theta_{\mathrm{bb}}}
\;+\;\beta\underbrace{\mathbb{E}\,\mathrm{KL}\Big(\mathrm{sm}\big(z_t(x)/\tau\big)\,\Big\Vert\,\mathrm{sm}\big(W_s\,\sigma\,\mathrm{sg}[\hat\Phi_s(x_{\mathrm{adv}})]+b_s\big)\Big)}_{\mathcal{L}_{\mathrm{hd}}\ \to\ \theta_{\mathrm{hd}}}\ }
\tag{2}$$

where $\mathrm{sm}$ is softmax and $\mathrm{sg}[\cdot]$ is stop-gradient. An optional third term (§3.6) is **off** in the champion.

Comparing (2) with (TRADES) term by term:

| | natural term | boundary term | anchor | space | parameters reached |
|---|---|---|---|---|---|
| TRADES | $\mathrm{CE}(f(x),y)$ | $\mathrm{KL}(f(x)\Vert f(x'))$ | own clean output (drifts) | logit | all |
| **ours** | $\beta\,\mathcal{L}_{\mathrm{hd}}$ | $\mathcal{L}_{\mathrm{dir}}$ | **frozen** teacher direction $\hat\Phi_t(x)$ | **feature** | **head only** / **backbone only** |

> **The method in one sentence.** Take the TRADES decomposition, split its two terms across disjoint parameter blocks, and replace the boundary term's self-anchor with the frozen feature *direction* of a naturally-trained teacher.

### 3.4 Three propositions

> ⚠ **ALL THREE ARE OBSOLETE FOR THE SHIPPED DESIGN (2026-08-31).** They were written for the
> *directional* objective and none survives the move to the raw-$\ell_2$ anchor:
> **Prop. 1 (head firewall)** describes a head-KD term that no longer exists — `featdir_freeze_head`
> is on and the head is not trained at all; **Prop. 2 (magnitude neutrality)** is a statement about
> $\nabla\lVert\hat\Phi_s-\hat\Phi_t\rVert^2$ being orthogonal to $\Phi_s$, which is **false**
> for $\lVert\Phi_s-\Phi_t\rVert^2$; **Prop. 3** is about the $k$-subspace knob, a dead axis.
> The paper's propositions are the ones in `writting_docs/method.md` §3.3, and they are different.
> Kept here as a record of the directional design, not as source material.



**Proposition 1 (head firewall).** *With $\lambda=0$, the parameter blocks carry disjoint objectives:*

$$\frac{\partial\mathcal{L}}{\partial\theta_{\mathrm{bb}}}=\frac{\partial\mathcal{L}_{\mathrm{dir}}}{\partial\theta_{\mathrm{bb}}},
\qquad
\frac{\partial\mathcal{L}}{\partial\theta_{\mathrm{hd}}}=\beta\frac{\partial\mathcal{L}_{\mathrm{hd}}}{\partial\theta_{\mathrm{hd}}} .$$

*Proof.* $\mathcal{L}_{\mathrm{dir}}$ contains no head parameters. In $\mathcal{L}_{\mathrm{hd}}$ the feature enters only through $\mathrm{sg}[\hat\Phi_s]$, whose Jacobian w.r.t. $\theta_{\mathrm{bb}}$ is zero by definition of stop-gradient. $\square$

Consequences: the backbone never sees a label, a teacher logit, or $W_t$ — it is trained **exclusively** by the boundary term. The head never sees the feature-matching objective. $W_t$ enters the computation graph only via $z_t/\tau$, which reaches $\theta_{\mathrm{hd}}$ alone. This is the formal content of *"inherit $\mathcal{F}$, never inherit $w$."*

**Proposition 2 (magnitude neutrality).** *The direction loss exerts zero force on the student's feature magnitude:*

$$\Big\langle \Phi_s,\ \nabla_{\Phi_s}\big\lVert\hat\Phi_s-\hat\Phi_t\big\rVert^2\Big\rangle\;=\;0 .$$

*Proof.* With $J=\partial\hat\Phi/\partial\Phi=\frac{1}{\lVert\Phi\rVert}(I-\hat\Phi\hat\Phi^\top)$ and $\lVert\hat\Phi_s-\hat\Phi_t\rVert^2=2-2\hat\Phi_s^\top\hat\Phi_t$,

$$\nabla_{\Phi_s}\mathcal{L}_{\mathrm{dir}}=J^\top(-2\hat\Phi_t)=-\frac{2}{\lVert\Phi_s\rVert}\big(I-\hat\Phi_s\hat\Phi_s^\top\big)\hat\Phi_t ,
\tag{3}$$

and $\Phi_s^\top(I-\hat\Phi_s\hat\Phi_s^\top)=\Phi_s^\top-\lVert\Phi_s\rVert\hat\Phi_s^\top=0$. $\square$

*Shorter proof.* $\mathcal{L}_{\mathrm{dir}}$ depends on $\Phi_s$ only through $\hat\Phi_s$, hence is
homogeneous of degree $0$; Euler's identity $x^\top\nabla f=k f$ with $k=0$ gives the claim directly.

Two readings of (3):
1. the update is a **pure rotation** of $\hat\Phi_s$ toward $\hat\Phi_t$ inside the tangent space of the sphere — the projector $(I-\hat\Phi_s\hat\Phi_s^\top)$ annihilates the radial component;
2. the factor $\lVert\Phi_s\rVert^{-1}$ is an **implicit per-sample weight**: large-norm (typically easy/confident) samples receive proportionally smaller gradient. No explicit re-weighting schedule is needed to obtain it.

**What Prop. 2 does *not* say.** The statement is about the gradient w.r.t. the *feature*, to first
order. It does **not** claim $\lVert\Phi_s(x)\rVert$ is invariant during training: parameter updates
couple samples, and weight decay, BN and AWP all act on the norm directly. The usable consequence is
narrower and is the one §4 needs — **no term of the objective ever pushes $\lVert\Phi_s\rVert$ toward
$\lVert\Phi_t\rVert$**, since the head also sees only $\hat\Phi_s$. The student's feature magnitude is
left as a free internal quantity rather than being bound to an unreachable teacher target.

**Proposition 3 ($Q$ is a rotation at $k=d$).** *If $k=d$ then $Q\in O(d)$ and $\lVert Q^\top v\rVert=\lVert v\rVert$, so (1) and (2) reduce to the plain objectives*

$$\mathcal{L}_{\mathrm{dir}}=\mathbb{E}\big\lVert\hat\Phi_s(x_{\mathrm{adv}})-\hat\Phi_t(x)\big\rVert^2=\mathbb{E}\big[\,2-2\cos\angle\big(\Phi_s(x_{\mathrm{adv}}),\Phi_t(x)\big)\big].$$

The champion uses $k=d=512$, i.e. **no subspace restriction**: the projection is a no-op and $\mathcal{L}_{\mathrm{dir}}$ is exactly a cosine loss. ($k<d$ is retained in the code as an ablation knob; see §5.)

### 3.5 On the head target

$z_t=\lVert\Phi_t\rVert\,W_t\hat\Phi_t+b_t$, hence

$$\frac{z_t}{\tau}\;=\;\frac{\lVert\Phi_t\rVert}{\tau}\,W_t\hat\Phi_t+\frac{b_t}{\tau},
\qquad \frac{\mathbb{E}\lVert\Phi_t\rVert}{\tau}\approx\frac{13}{16}\approx0.81 .$$

At $\tau=16$ the soft target is therefore, up to a constant close to $1$, the teacher's **cosine** response — the temperature is not an arbitrary smoothing constant but the scale that converts the teacher's logits into a magnitude-free target consistent with $\mathcal{L}_{\mathrm{dir}}$.

### 3.6 Optional consistency term (off in the champion)

$$\mathcal{L}_{\mathrm{cons}}=\mathrm{KL}\Big(\mathrm{sm}\big(z_s(x)\big)\,\Big\Vert\,\mathrm{sm}\big(z_s(x_{\mathrm{adv}})\big)\Big),
\qquad \mathcal{L}\mathrel{+}= a(e)\,\lambda\,\mathcal{L}_{\mathrm{cons}},\quad a(e)=(e/E)^2 .$$

This is **literally the TRADES boundary term**. It is set to $\lambda=0$; see §5 for why.

### 3.7 Weight averaging and schedule

$$\bar\theta^{(j+1)}=d_e\,\bar\theta^{(j)}+(1-d_e)\,\theta^{(j+1)},\qquad
d_e=\kappa+(1-\kappa)\Big(\frac{e}{E}\Big)^{2},\quad\kappa=0.999,$$

updated per step for $e\ge e_{\mathrm{wa}}$; **all reported metrics are computed with $\bar\theta$.** The learning rate follows one-cycle and is frozen at its current value for $e\ge e_{\mathrm{fr}}$:

$$\eta_e=\begin{cases}\mathrm{OneCycle}(e)&e<e_{\mathrm{fr}}\\ \mathrm{OneCycle}(e_{\mathrm{fr}})&e\ge e_{\mathrm{fr}}\end{cases}$$

Note $d_e\to1$ as $e\to E$: the average freezes at the end of training. The EMA horizon is $\approx(1-d_e)^{-1}\approx10^3$ steps $\approx2.6$ epochs early, growing to $\gg E$ at the end.

### 3.8 Optional: AWP (bolt-on, not part of the claim)

$$\nu^\star=\arg\max_{\lVert\nu_l\rVert\le\gamma\lVert\theta_l\rVert\ \forall l}\ \mathcal{L}(\theta+\nu),\qquad
\theta\leftarrow\theta-\eta\,\nabla_\theta\mathcal{L}(\theta+\nu)\big|_{\theta+\nu},$$

layer-wise normalized, one ascent step, active for $e\ge e_{\mathrm{awp}}$. This is Wu et al. verbatim; it contributes to the number but not to the mechanism.

### 3.9 Full algorithm

```
input: natural teacher θ_t, epochs E, radius ε_tr, steps m
θ ← θ_t                                   # backbone AND head
θ̄ ← θ
for e = 0 … E-1:
  for (x,y) in loader:
    φ_t ← normalize(Φ_t(x));  z_t ← teacher logits          # no_grad
    x_adv ← PGD_m maximizing ‖Qᵀ(normalize(Φ_s(·)) − φ_t)‖²  # eq (1)
    L ← ‖Qᵀ(normalize(Φ_s(x_adv)) − φ_t)‖²                   # → backbone
        + β·KL( sm(z_t/τ) ‖ sm(head(σ·sg[normalize(Φ_s(x_adv))])) )   # → head
    (optional AWP wrap)  θ ← θ − η ∇L
    forward Φ_s(x) once to refresh BN running statistics
    if e ≥ e_wa:  θ̄ ← d_e θ̄ + (1−d_e) θ
    if e <  e_fr: lr_scheduler.step()
evaluate θ̄
```

### 3.10 Champion hyper-parameters (ResNet-18 — **identical on CIFAR-10 and CIFAR-100**)

$$E=100,\ \ \text{AdamW},\ \eta_{\max}=0.021,\ \ \beta=1,\ \ \tau=16,\ \ \sigma=1,\ \ k=d=512,\ \ \lambda=0,$$
$$m=10,\ \ \alpha=2/255,\ \ \varepsilon_{\mathrm{tr}}=8.8/255,\ \ \varepsilon=8/255,\ \ \kappa=0.999,\ \ e_{\mathrm{wa}}=0.2E,\ \ e_{\mathrm{fr}}=0.65E,$$
$$\text{AWP: } \gamma=0.005,\ e_{\mathrm{awp}}=10,\qquad \text{sensitivity-matched }\varepsilon\text{: } p=1,\ w_{\lo}=0.5,\ w_{\hi}=1.5 .$$

`config/CIFAR10/featdir_champ200_angeps.yaml` and `config/CIFAR100/featdir_champ200_angeps.yaml`
differ **only** in `dataset` and the teacher checkpoint path — every knob above is shared. There is
**no per-dataset tuning**, which is worth stating explicitly in the paper: the recipe was selected
on CIFAR-100 and transferred to CIFAR-10 unchanged.

---

## 4. What the objective is *asking for*, and why it cannot be satisfied

> ⚠ **Partly superseded by §2b (2026-08-02).** The argument below concludes that magnitude must be
> dropped because it is unreachable. `featdir_champ200_rawfeat` tested exactly that and came back a
> **robustness tie** (PGD 34.94 vs 34.94, AA $-0.12$): keeping the magnitude costs nothing robust,
> only $-0.95$ clean. Claim 1 in §2b reaches the right prediction without invoking unreachability at
> all. Read §4 as the origin of the free-head argument (point 2, which survives and is now Claim 2),
> not as a justification for normalization.

The target $\hat\Phi_t(x)$ is computed (i) on the **clean** input and (ii) by a **natural** teacher, while the constraint is imposed at $x_{\mathrm{adv}}$. Decompose the teacher's representation into $\gamma$-robust and useful-non-robust parts,

$$\Phi_t = \Phi_t^{(r)} + \Phi_t^{(n)} .$$

By definition $\Phi_t^{(n)}$ flips under some $\delta\in\mathcal{B}(0,\varepsilon)$; no student can reproduce it at $x_{\mathrm{adv}}$ for all $x$. Hence

> $\mathcal{L}_{\mathrm{dir}}$ **has a strictly positive floor** on the non-robust component: the objective demands that non-robust features be made robust.

This is a design property, not a defect, and three things follow.

1. **Magnitude must be dropped.** $\lVert\Phi_t\rVert$ is largely built from $\Phi_t^{(n)}$ (teacher confidence tracks the non-robust bulk), so matching magnitude imposes an unreachable target, leaving a residual gradient that is absorbed by memorization. The unit direction is the reachable part of the request; Prop. 2 makes the neutrality exact.
2. **The head must stay free.** The residual the backbone *cannot* clear has to be discounted somewhere. The only place is $W_s$, which can down-weight the coordinates that failed to become robust. **The free head is the release valve of an unsatisfiable backbone target.** Freezing either its direction (gain-only head) or its magnitude (cosine head) closes the valve.
3. **$\mathcal{L}_{\mathrm{dir}}$ doubles as a smoothness term.** The anchor is adversary-independent, so matching it at $x_{\mathrm{adv}}$ implicitly forces $\hat\Phi_s(x_{\mathrm{adv}})\approx\hat\Phi_s(x)$. This is why the loss is insensitive to the exact form of the inner attack, whereas KL-shaped losses require their matched adversary.

---

## 5. Predictions, and how the ablations land

All numbers CIFAR-100 / ResNet-18, single seed unless noted. Noise floor from this project's paired 3-seed runs: $\approx\pm0.3$–$0.4$.

> ⚠ **Regime warning (2026-08-01).** The head/normalization rows below were measured in the
> **3-step, no-WA, $k{=}512$, 2026-07-13 era** — and the cosine-head cells not even under
> `feat_direction` (its `head_from_feat` did not exist on `resnet_zcos` until 2026-08-01, so every
> earlier `coshead` number comes from the baseline-KL / `madry_at` methods). That regime sits at
> PGD $\approx29$–$31$; the shipped recipe sits at PGD $\approx34$. **They are not comparable, and
> these rows must not be cited as evidence about the shipped method until the re-run lands.** A
> 12-cell grid at a single fixed regime (50ep / 10-step / WA / $k{=}512$ / $\lambda{=}0$ / no AWP /
> `clean_200ep` teacher) — {teacher feature norm} $\times$ {teacher head norm} $\times$ {student
> head: free, cosine, gain} — is in flight to replace them. Rows marked ⏳ below are the ones at
> stake.

| prediction (from §2/§4) | test | measurement | verdict |
|---|---|---|---|
| $w$ must be re-solved ⇒ freezing **either** channel of the head hurts, **symmetrically** | head $2\times2$ | free/free $H=41.77$; cosine head $40.01$; gain-only head $39.99$ | ⏳ symmetric $-1.8$ **but stale regime** |
| feature normalization acts on $\mathcal{F}$, not $w$ ⇒ neutral by itself | hard-CE AT | no FN $58.37/27.50$ vs FN $58.45/27.23$ | ⏳ inside floor, different method |
| weight normalization removes the re-weighting channel ⇒ **harmful** (opposite of HE) | hard-CE AT | FN + cosine head $57.82/25.75$ | ⏳ $-1.75$ PGD, different method |
| teacher-side normalization is second-order (the student overwrites $w$ anyway) | `normfeat_target`, champion regime | $62.06/33.44/28.03$ vs $62.75/33.96/28.41$ | ✅ tie / slight loss |
| the re-weighting channel is actually used | $\lVert w_c\rVert$ during AT | $1.85\to9.5$ ($5\times$), class spread $1.37\times$ | ✅ |
| $\gamma$-discounting lives in $w$, not in the choice of features ⇒ **which** features are matched is second-order | $Q$ from random / teacher span / natural-PCA / robust-PCA / oracle | all tie within $0.07$–$0.3$ | ✅ |
| …including a subspace chosen by **directly measured** per-coordinate $\gamma$ | `decorr`: keep the $k$ coordinates minimizing $\mathbb{E}\lvert\Phi_t(x)_j-\Phi_t(x_{\mathrm{adv}})_j\rvert\cdot e^{-\beta\,\mathrm{need}_j}$, $k=350$, matched regime | decorr $62.19/33.34/28.24$ vs random $62.75/\mathbf{33.96}/\mathbf{28.41}$ | ✅ (random wins if anything) |
| …and **how many** is second-order too | $k=350$ vs $k=512$ under the final recipe | $+0.31$ AA (CIFAR-100), tie (CIFAR-10) | ✅ (but see ⚠) |
| $\mathcal{L}_{\mathrm{dir}}$ already occupies the boundary slot ⇒ adding TRADES's own boundary term is **redundant** | $\lambda$ sweep | $\lambda\in\{0.3,\dots,100\}$ all $\le\lambda=0$ at $k=350$; champion $\lambda=0$ | ✅ |
| a $w$-level and an $\mathcal{F}$-level constraint together over-constrain | baseline KL $+$ dir term ("plugin") | $65.44/29.15$, PGD **below both parents** | ✅ |

⚠ **Open discrepancy.** In the 3-step regime $k$ was a *first-order* effect ($k{=}350$: $30.52$ PGD vs $k{=}512$: $28.91$). It collapses to second-order only once 10-step / WA / AWP / long schedule are in. The frame predicts the latter and not the former; the reason for the regime dependence is unexplained. This is the weakest joint in the argument and should be stated as such.

**Not yet run — the decisive test.** The claim "the head must be free *because the teacher supplies a $\gamma$-blind $w_t$*" predicts an **interaction**: with a *robust* teacher, freezing the head should cost much less.

| | free head | cosine head |
|---|---|---|
| natural teacher | $41.77$ ✔ | $40.01$ ✔ |
| **robust teacher** | ? | ? |

Predicted: gap $-1.8\to\approx0$. No interaction ⇒ the causal story in §4.2 is wrong.

*This is a diagnostic, not a change to the method* — the shipped method uses a natural teacher (P1).
The robust teacher is an instrument for isolating the mechanism and appears in the paper as two
ablation rows. It requires no new training: `CIFAR100/checkpoint/at_ce_freehead/madry_at_last.pkl`
(clean-init AT, $58.45/27.23$; shares the coordinate basis, so it is usable as both teacher and
initialization) and `CIFAR100/checkpoint/at_teacher/madry_at_last.pkl` (scratch AT) are on disk.
Prefer the clean-init one: a scratch-AT model lives in a different basis, which invalidated the
earlier robust-PCA cell for the same reason.

---

## 6. Honest limitations

1. **The TRADES guarantee does not transfer.** Thm. 3.1 bounds $R_{\mathrm{rob}}-R^*_{\mathrm{nat}}$ when *both* terms use the same $f$. Replacing the boundary anchor with an external frozen teacher voids the bound. Our objective is TRADES-**shaped**, not TRADES-**justified**.
2. **We use only Ilyas §2** (the $C=(\mathcal{F},w,b)$ decomposition and $\rho/\gamma$). Their Theorems 1–3 are binary-Gaussian/linear; an earlier attempt to route the $k$-story through the $\Sigma\to I$ argument was falsified by the $k=350$ vs $k=512$ ablation. Do not re-import it.
3. **$w_c$ is per-class, Ilyas's $w$ is per-feature.** The mapping (row $w_c$ = weighting of the $d$ features for class $c$) is natural but must be stated, not assumed.
4. **The head target is a soft teacher distribution, not hard CE**, so its correspondence with $\psi^{-1}(R_\phi-R^*_\phi)$ in Thm. 3.1 is loose.
5. **Feature-space (rather than logit-space) boundary term** is justified here only by reachability (§4), not by either source paper.
6. **Single seed** on essentially every cell above.
7. **WA, AWP and the enlarged $\varepsilon_{\mathrm{tr}}$ are off-the-shelf** and account for a large share of the final number ($+0.82$ and $+0.56$ AA respectively).

---

## 7. Current numbers

### 7.0 Champion — sensitivity-matched $\varepsilon$ on both datasets (2026-08-04)

`featdir_champ200_angeps`, ResNet-18, seed 0, `last` checkpoint. Same recipe on both datasets
(§3.10), no re-tuning.

| | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **CIFAR-100, ours + angeps** | **62.17** | 34.77 | **30.92** | 28.59 | **39.17** |
| CIFAR-100, ours ($p=0$) | 60.74 | 34.94 | 30.53 | 28.69 | 38.97 |
| CIFAR-100, ADR-full | 57.36 | 34.92 | 30.62 | 28.50 | 38.08 |
| **CIFAR-10, ours + angeps** | **84.66** | 56.74 | **53.94** | 51.87 | **64.33** |
| CIFAR-10, ours ($p=0$) | 82.52 | 57.20 | 53.74 | 51.89 | 63.71 |
| CIFAR-10, ADR-full | 83.26 | — | — | 51.18 | 63.39 |
| CIFAR-10, CURE | 86.76 | 54.92 | 52.48 | 49.69 | 63.19 |

NRR and CW are first place in the full comparison tables on **both** datasets
(`comparison_resnet18_cifar10.md`, `comparison_resnet18_cifar100.md`).

**The signature is the same on both datasets, and the claim must be stated as such:**

> **AA is a tie** (CIFAR-100 $-0.10$, CIFAR-10 $-0.02$, both inside the observed noise band
> $28.46$–$28.71$), and what is won is **clean $+1.4$ to $+2.1$**, with CW slightly up and PGD
> slightly down.

So the honest sentence is *"recovers natural accuracy at matched robustness"*, **not** "improves
AA". The CW gain compresses sharply on the way to AA ($+0.39\to-0.10$ and $+0.20\to-0.02$), which
is itself the reminder that CW margins do not transfer to AA — only AA arbitrates.

Two things this does *not* yet have: **seed 0 only** on both datasets, and the ablation table mixes
schedules (the base row is 50ep, the $+$WA rows are 100ep). Both are listed in §6.

A stacked view on CIFAR-100 — the allocation helps at every level of the stack, always as
clean $\uparrow$ *and* CW $\uparrow$ together:

| stack | clean | PGD-20 | CW | AA | NRR |
|---|---|---|---|---|---|
| base 50ep | 62.61 → **64.59** | 29.16 → 28.77 | 26.63 → **27.05** | — | — |
| $+$WA | 60.73 → **62.25** | 33.32 → 32.49 | 29.60 → **30.32** | 28.08 → 28.13 | 38.40 → **38.75** |
| $+$WA$+$AWP | 60.74 → **62.17** | 34.94 → 34.77 | 30.53 → **30.92** | 28.69 → 28.59 | 38.97 → **39.17** |

This is also the **first intervention to transfer from the 50ep screening regime to the champion
recipe** — the raw/raw normalization cell and AWP both won at 50ep and then failed to carry — and
the first to move CW outside the $26.4$–$26.8$ band that every *target-geometry* manipulation was
trapped in. Consistent with the working rule: **touch the target and clean moves; touch the attack
and robustness moves.**

### 7.1 The preceding champion — ADR beaten on both axes (2026-08-01)

| model | clean | PGD-20 | CW | AA | H(pgd) | H(cw) | NRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ours, `featdir_champ200_100ep`** | **60.74** | **34.94** | **30.53** | **28.69** | **44.36** | **40.64** | **38.97** |
| ADR-full (strongest baseline) | 57.36 | 34.92 | 30.62 | 28.50 | 43.41 | 39.93 | 38.08 |
| other-server champion (target) | 60.67 | — | — | 28.42 | — | — | 38.71 |
| our previous best (100ep+AWP, $\varepsilon_{\mathrm{tr}}$10) | 60.04 | 34.36 | 29.31 | 27.36 | 43.71 | 39.39 | 37.59 |
| 50ep champion | 62.75 | 33.96 | 28.41 | 26.29 | 44.07 | 39.11 | 37.06 |

**This is the first configuration that beats ADR-full on AA (+0.19) while also leading clean by
+3.38.** PGD-20 +0.02 and CW −0.09 are ties. Every derived metric is a project record: H(pgd)
44.36, H(cw) 40.64, NRR 38.97 (vs ADR 38.08). It also slightly exceeds the other-server champion
it was reproducing (AA 28.42 → 28.69).

The P1/P2 reading is now unqualified: **both axes at or above the strongest baseline, obtained
without an adversarially-trained teacher.** The trade-off framing is no longer load-bearing — it
is a strict improvement plus a cheaper teacher.

### The recipe

`config/CIFAR100/featdir_champ200_100ep.yaml`, CIFAR-100 / ResNet-18, seed 0:

| | |
|---|---|
| teacher | `clean_200ep` — plain natural training, 200 epochs, SGD 0.1 + one-cycle, no mixup, **77.59** clean |
| init | student backbone **and** head from the same checkpoint (§3.1) |
| optimizer | AdamW, $\eta_{\max}=0.021$, one-cycle |
| epochs | 100, **`freeze_lr_epoch: 0.65`** → LR frozen at epoch 65 (at 0.0167, i.e. 79 % of peak) |
| WA | on, $\kappa=0.999$, **`wa_start: 0.2`** → shadow starts at epoch 20; **the WA average is what is evaluated** |
| attack | PGD-10, $\alpha=2/255$, **$\varepsilon_{\mathrm{tr}}=8.8/255$** (= 8/255 × 1.1); evaluated at 8/255 |
| loss | $k=512$ (projection is a no-op, §3.4 Prop. 3), $\lambda=0$, $\beta=1$, $\tau=16$ |
| AWP | proxy, $\gamma=0.005$, warmup 10 |

```bash
python main.py --config_name featdir_champ200_100ep.yaml --dataset CIFAR100 --seed 0
```

### What is *not* yet known about it

Five things moved at once relative to the previous best (60.04 / 27.36): the teacher (300ep → the
new 200ep), `freeze_lr_epoch`, `wa_start`, $\varepsilon_{\mathrm{tr}}$ (10 → 8.8/255), and $\lambda$
(1.5 → 0). **The +1.33 AA is not yet attributed to any of them.** `freeze_lr_epoch` is the leading
suspect: it is the only one that had no implementation at all before today, and freezing the LR at
79 % of peak for the last 35 epochs turns the tail into a constant-LR phase that WA then averages
over — textbook SWA, which is exactly the kind of thing that moves AA. Single seed.

---

# 260825 — Why the anchor buys **clean accuracy**

*Written 2026-08-25, argument revised twice and reordered 2026-08-26. This is the section
theory_v1 T.8 asks for. Closed-form numbers reproduce with `python scripts/theory_spectrum.py`;
measured numbers with `python scripts/diag_retention_within_family.py`.*

## 8.0 The thesis, and the ladder that states it

T.8's verdict on the existing theory is correct and blunt: **Theorem 1 explains why the method does
not fail, not why it works.** Inside T.1's model the anchored optimum equals plain AT's classifier,
so the headline — clean $57.36\to62.17$ at AA $28.50\to28.59$ — is not merely unexplained there, it
is impossible there.

The thesis is the intuitive one, and it is measurable:

> **Clean accuracy is how much of the teacher's representation the student still uses.** The
> method's content is that it can keep almost all of it without paying robustness.

Every cell below starts from the **same** teacher checkpoint, runs the same 100-epoch recipe, and
differs in one term, so initialization and schedule are held fixed and all of them share the
teacher's coordinate basis — which is what makes $\cos(\hat\Phi_s(x),\hat\Phi_t(x))$ a meaningful
number. CIFAR-100, full test set, each cell attacked by its own PGD-10 CE at $\varepsilon=8/255$;
AA quoted from §7/T.3–T.4.

| how the clean teacher is used | clean | PGD-10 | AA | $\cos$ vs teacher |
|---|---:|---:|---:|---:|
| not at all — ADR self-anchors | 57.37 | 35.18 | 28.50 | *(different net)* |
| as an **initialization only**, then label-CE AT | 58.38 | 28.28 | — | **0.4703** |
| through its **logits** (`nofeat`) | 58.94 | 35.45 | 28.71 | **0.0968** |
| through its **features** (champion $p{=}0$) | 60.76 | 35.17 | 28.69 | **0.8245** |
| + sensitivity-matched $\varepsilon$ (champion) | **62.16** | 35.06 | 28.59 | **0.8345** |
| the teacher itself | 77.65 | 0.00 | ≈0 | 1.0000 |

**Read the two columns against each other.** Among the four cells trained at the shipped regime,
"how well the clean teacher is used" is a monotone axis, $\cos$ vs teacher is its gauge, and
**robustness is constant along the whole of it** — PGD $35.06$–$35.45$, a range of $0.39$, AA within
$\pm0.11$ of $28.60$.

**Warm-starting at the teacher is not what keeps a student there.** Rows two and three both begin
*at* the teacher checkpoint and both walk away from it: label-CE adversarial training ends at
$\cos=0.470$, logit distillation at $\cos=0.097$. Only the feature anchor stays ($0.82$–$0.83$).
Both feature maps are non-negative — post-ReLU global pool — so a cosine of $0.097$ means
near-disjoint support: the logit-distilled student ends up using *different coordinates* from the
teacher it was distilled from. **A logit target names the teacher without keeping the student inside
it**, and an initialization does not hold on its own. That is the whole gap between "uses a clean
teacher" and "uses it well," and between the two regime-matched cells it is worth $+1.8$ clean at
$-0.02$ AA.

⚠ *The AT clean-init row is a `cos` control, not a clean-accuracy comparison: that checkpoint is from
the 2026-07 3-step / no-WA era (§5's regime warning) and sits at PGD $28.28$ against $\approx35$ for
every other row. It answers "does starting at the teacher keep you there" and nothing else.*

**Two questions follow, and only the second needs theory.** *Why does keeping the teacher buy clean?*
— it is what the ladder measures, and the teacher is at $77.65$. *Why is keeping a **non-robust**
teacher free on robustness?* — that is the surprising half, it is the objection ADR's self-anchoring
exists to avoid, and §8.1 is the answer. §8.2 then asks why plain AT does not simply do this.

## 8.1 Why it is free: the anchor is evaluated at the clean point

Two results already in this file, read together.

**Theorem 1 (T.1): the adversarial term is an $\ell_1$ penalty on the non-robust coefficient.** With
$\Phi_s=ax_1+b\,m(z)$ and the inner maximum taken exactly,
$J(a,b)=\mathbb{E}[R^2]+2\varepsilon\lvert b\rvert\mathbb{E}\lvert R\rvert+\varepsilon^2b^2$ is
kinked at $b=0$ with subgradient jump $2\varepsilon\mathbb{E}\lvert R\rvert$, so past a threshold
$\varepsilon_0$ the minimizer has $b^\star=0$ **exactly** — and by Cor. 2 the solution is then
$\varepsilon$-independent. The teacher supplies the *value* to be matched; the inner maximization
discards the route the teacher took to it. The non-robust component is not inherited, because it
cannot be reproduced at $x_{\mathrm{adv}}$ and the $\ell_1$ kink zeroes it rather than shrinking it.

**Proposition 2 (T.2b): the one term is fidelity plus stability.** $L\le F+O\le3L$ by the triangle
inequality, with $F=\lVert\Phi_s(x)-\Phi_t(x)\rVert$ and $O$ the oscillation of the student's own
map on the ball. $\Phi_t$ is evaluated at $x$ and nowhere else, so **the teacher's instability never
enters the objective**. The student is not asked to imitate the teacher under attack; it is asked to
hold the teacher's clean value while under attack — which is exactly what the teacher cannot do.

Together: $F$ is what keeps the student in the teacher's coordinates (the ladder's $\cos$ column),
$O$ is what makes it stable, one term delivers both, and neither asks the student to reproduce
anything the teacher does *off* the clean point. That is why the clean axis and the robust axis move
independently — the paper's standing observation that every anchor-side intervention lands at an AA
tie and moves only clean (T.4) is this decomposition seen in the ablation table.

## 8.2 Why plain AT does not do this: stability by deletion

If keeping the teacher's geometry is free, why does adversarial training give it up? Because it is
never asked to keep it, and the cheapest route to stability is to stop reading the volatile
directions. The following is a model, and it is used for the shape of the answer, not its size.

**Setup.** $x\mid y\sim\mathcal N(\eta y,I)$ on $\mathbb{R}^D$, $\eta\ge0$; $\ell_\infty$ adversary
of radius $\varepsilon$ per coordinate. Coordinate $j$ has **reliability** $\eta_j$, and
$\eta_j/\varepsilon$ is the only thing that matters. For a linear readout $c^\top x$,
$M_c=c^\top\eta/\lVert c\rVert_2$ and $M_r=(c^\top\eta-\varepsilon\lVert c\rVert_1)/\lVert c\rVert_2$
— the $\ell_\infty$/$\ell_1$ duality T.5 already uses.

**Proposition A1 (AT is a hard threshold).** $\arg\max_c M_r\propto(\eta-\varepsilon)_+$.
*Proof.* Signs may be taken non-negative; then $c^\top(\eta-\varepsilon)\le c^\top(\eta-\varepsilon)_+\le\lVert c\rVert_2\lVert(\eta-\varepsilon)_+\rVert_2$
by Cauchy–Schwarz, with equality iff $c\propto(\eta-\varepsilon)_+$. $\square$
Not an artifact of optimizing accuracy instead of a surrogate: minimizing the adversarial **logistic**
loss returns the same solution (support $64$ vs $64$, $\mathrm{corr}=1.0000$, identical accuracies).

**So every coordinate with $\eta_j\le\varepsilon$ gets weight exactly zero, however close $\eta_j$ is
to $\varepsilon$.** Usefulness is continuous; the robust treatment of it is not.

**Proposition A2 (what the discontinuity costs).** Restoring weight to a deleted coordinate trades at
$$\frac{dM_c}{-dM_r}=\frac{\eta_j}{\varepsilon-\eta_j}\ \xrightarrow[\eta_j\uparrow\varepsilon]{}\ \infty .$$
Verified against finite differences (predicted / measured: $67.28/67.22$ at $\eta_j/\varepsilon=0.986$,
$15.30/15.30$ at $0.943$, $4.01/4.01$ at $0.812$). Relaxing the threshold to
$c_\lambda=(\eta-\lambda)_+$ traces a frontier on which, just below the AT solution, clean rises
several points for a robustness cost inside any realistic noise band ($\lambda/\varepsilon=0.9$:
$+3.71$ clean, $-0.20$ robust; the rate is $\approx18$ across four unrelated spectra, while the
*size* of the gain varies by two orders of magnitude — the rate is threshold geometry, the size is
how much reliability mass sits near $\varepsilon$).

**Why the existing theory could not see any of this.** T.1's spectrum has two atoms —
$\eta_1/\varepsilon=\infty$ and bulk at $1/2$ — and **no mass near $1$**, where the entire effect
lives. On that spectrum, relaxing the threshold changes nothing at all until it crosses $0.5$, at
which point the whole bulk returns at once and robustness collapses 22 points:

| $\lambda/\varepsilon$ | 1.00 | 0.90 | 0.70 | 0.50 | 0.00 |
|---|---:|---:|---:|---:|---:|
| clean | 69.09 | 69.09 | 69.09 | 69.09 | 77.40 |
| robust | 67.31 | 67.31 | 67.31 | 67.31 | 45.04 |

**T.1 rules out the paper's effect by construction, and so does any two-atom model.** That is a
property of the data model, not of the objective, and it is the precise reason Theorem 1 finds
anchoring and AT to be the same classifier. This is the diagnosis T.8 asked for.

⚠ **Proposition C, and the limit of every re-weighting account.** For a linear readout,
$$M_c(c)-M_r(c)=\varepsilon\,\frac{\lVert c\rVert_1}{\lVert c\rVert_2}\in[\varepsilon,\varepsilon\sqrt D],$$
exactly: reading a coordinate and being exposed on it are the *same* parameter, so deletion is the
only gap-reducing move a linear map has. Consequently no re-allocation of feature weight reproduces
the measured signature — across four spectra $\times$ six radii the anchored objective's own optimum
trades clean for robust at $0.46$–$2.6$, against a measured rate $\ge12$. **A nonlinear map escapes
this** because its exposure is $O$, which is not a function of how many coordinates it reads: it can
depend on an attackable coordinate and still be locally constant in it. That is what §8.1's $F+O$
asks for and what adversarial CE does not.

## 8.3 The full within-family measurement

The ladder of §8.0 in full; same protocol; effective rank is the participation ratio of the **unit**
feature covariance (on raw features it is confounded by the unconstrained norm under a normalized
head); probe is a linear probe from frozen clean features to labels, 5 000 fit / 5 000 scored. Log:
`results/CIFAR100/retention_within_family_20260826.log`.

| cell | clean | PGD-10 | feature angle | $\cos$ vs teacher | eff. rank | label probe |
|---|---:|---:|---:|---:|---:|---:|
| teacher (natural) | 77.65 | 0.01 | 63.8° | 1.0000 | 62.4 | 77.14 |
| champion, $p{=}1$ | **62.16** | 35.08 | 14.6° | **0.8345** | 32.8 | 62.14 |
| `alpha1` (head grad reaches backbone) | 61.03 | 35.27 | 13.1° | 0.8245 | 31.5 | 60.66 |
| champion, $p{=}0$ | 60.76 | 35.17 | 13.1° | 0.8245 | 31.5 | 60.32 |
| `rawfeat` (raw target) | 59.78 | 35.10 | 12.3° | 0.8178 | 30.7 | 58.56 |
| `nofeat` (**anchor deleted**) | 58.94 | 35.45 | 6.3° | **0.0968** | **2.1** | 57.76 |
| AT clean-init ⚠ *other regime* | 58.38 | 28.28 | 15.7° | **0.4703** | 48.1 | 51.52 |

*Every cell reproduces its documented clean accuracy to within $0.02$ ($62.17/61.03/60.74/59.79/58.92$
in §7 and T.3–T.4); the teacher's $63.8°$ rotation and $\times2.45$ norm inflation reproduce §1's
independently measured $62°$ and $\times2.47$; and the champion's $\cos$ comes out $0.8345$, matching
T.3's $0.8345$ exactly.*

Two geometry observables order the anchored cells exactly as clean accuracy does — $\cos$
$0.8178<0.8245\approx0.8245<0.8345$ and effective rank $30.7<31.5=31.5<32.8$ — while robustness does
not move. This reproduces T.3's 17-cell correlation and adds the column T.3 never had.

⚠ **The relation is an ordering, not a dose–response.** Among the anchored cells a $\cos$ change of
$0.017$ accompanies $+2.4$ clean; the $0.73$ collapse to `nofeat` accompanies only $-1.8$. A
representation with essentially nothing in common with the teacher still reaches $58.94$ clean and
$35.34$ PGD. So the anchor is demonstrably what holds the representation in the teacher's geometry,
and holding it is worth clean accuracy at no robustness cost — but §8.2's model does not predict the
*size* of that payment and must not be quoted as if it did. Effective rank is variance concentration,
not information dimension: `nofeat` probes at $57.76$ with a participation ratio of $2.1$, so its
class information lives in low-variance directions.

**A cross-method comparison was also run and is reported only as context.** Against ADR + WA + AWP
(clean $57.37$, PGD $35.18$, angle $15.1°$, CKA vs teacher $0.537$ against ours $0.619$, label probe
$55.12$ against ours $62.14$; `results/CIFAR100/delete_vs_stabilize_20260826.log`) our backbone
retains more of the teacher's clean geometry at matched oscillation. **It should not be used as
evidence for the mechanism**: our student is warm-started *at* the teacher and ADR is not, so higher
alignment is what initialization alone predicts, and the label probe lands within noise of each
model's own clean accuracy ($62.14/62.16$, $55.12/57.37$), so on that axis it re-measures the
headline. What it does add cleanly is that **the head is not where the gap lives** — consistent with
the head ablation, where refitting the head moves clean by $0.00$ ($62.72/62.77/62.70$).

## 8.4 Predictions

**8.4-1 (reliability mass near the threshold).** On the teacher, the histogram of
(per-coordinate class signal)/(per-coordinate attack displacement) should be unimodal across $1$
rather than two-atom. Decides how *large* the available gain is, not whether the mechanism is right.
Checkpoints on disk.

**8.4-2 (deletion is cheap for CE and forbidden by $F$).** Sweep a label-CE weight $\alpha$ on the
backbone: clean should fall monotonically at roughly constant AA as CE re-opens the deletion route.
One point already exists in the right direction and on the right axis — `alpha1`, clean $-1.14$ at
AA $28.59\to28.58$, with $\cos$ falling $0.8345\to0.8245$.

**8.4-3 (a second AT baseline). ✅ DONE 2026-08-26 — confirmed.** Clean-initialized label-CE AT
(`at_ce_freehead`) shares the teacher's basis, so its retention is not confounded by warm start. It
ends at $\cos=0.470$ against the anchor's $0.82$–$0.83$: **the initialization does not hold a student
in the teacher's coordinates; the anchor term does.** ⚠ That checkpoint is from a different (3-step,
no-WA) regime, so only its $\cos$ is usable. A regime-matched label-CE AT run started from the same
teacher would remove the last caveat and is one training run.

**8.4-4 (the trajectory).** $(\text{clean},\text{AA})$ along the finetune from the teacher should
trace a one-dimensional curve with $\cos$ falling monotonically. Partial support: $50\to100$ epochs
costs the directional design $1.86$ AA in the bare regime (T.6).

## 8.5 What is assumed, and what is not claimed

**Assumed.** §8.2 only: Gaussian coordinates with diagonal covariance, a linear readout, a
per-coordinate $\ell_\infty$ adversary — the setting T.1 and §2b already use, and used here mostly
to derive a *negative* result (Prop. C), which is the direction in which strong assumptions are
safe. §8.1 assumes nothing beyond Theorem 1's model and the triangle inequality; §8.0 and §8.3 are
measurements.

**Not claimed.**

1. **A quantitative prediction of the clean gain.** §8.2 gives the shape and sign of the trade, not
   its size; §8.3's caveat says why the measured relation cannot carry one either.
2. **That the anchored objective provably reaches the favourable operating point.** In the linear
   model it does not — it overshoots toward the teacher and pays roughly one for one (§8.2). Where a
   run lands is a question about the recipe and about nonlinear capacity the model does not
   represent.
3. **That freezing the head is the mechanism.** Head refit moves clean by $0.00$. ⚠ **This
   contradicts §2b Claim 2 ("the head must be re-solved"), the title of theory_v1 T.2, and T.2b's
   *Scope* sentence, which says the head is "fitted rather than inherited in the shipped recipe" —
   the shipped recipe inherits it.** Reconcile before printing; the surviving head statement is
   Prop. 0 / `featdir_alpha`$=0$ (routing).
4. **Any fit to the CIFAR numbers.** An earlier reverse-engineering matched to $0.3$ points only in
   a parametrization where the natural teacher's robust accuracy is at chance rather than near zero,
   and broke by $8$–$17$ points once that endpoint was corrected.
5. **That AT provably deletes in a real network.** A1 is about a robust-optimal linear readout.
   §8.3 measures the consequence inside our own family, where the objective is the only thing that
   changes, and 8.4-3 adds a label-CE AT run started from the same teacher ($\cos=0.470$) — but that
   run is from another regime, so the AT side of the comparison is still one regime-matched run
   short.

**One-line summary.** *Clean accuracy is how much of the teacher's representation the student still
uses. Starting at the teacher does not keep you there — label-CE AT drifts to $\cos=0.47$ and logit
distillation to $0.10$ — and only the feature anchor stays inside it ($\cos=0.83$); the ladder from
the logit route to the anchor is worth $3.2$ clean points at $0.39$ PGD. It is free because the anchor is evaluated at the clean point only, so the
teacher's instability never enters the objective; and it is available because adversarial training,
asked only that the decision survive the ball, buys stability the cheap way — by deleting the
directions the attack can move.*

---

# 260828 — Teacher training length is a trade-off dial, and the mechanism is class separation

*CIFAR-100, base regime (50 epochs, no WA, no AWP, $\varepsilon=8/255$), L2 target, head not
trained. Five student runs differing in **one thing only**: which natural teacher checkpoint they
anchor to and initialize from. Logs `results/CIFAR100/teacher_ladder_{separation,students}_20260828.log`;
reproduce with `scripts/diag_teacher_collapse.py` and `scripts/diag_teacher_ladder_students.py`.*

## 9.0 The result

| teacher | T.clean | T.Sw/Sb | **S.clean** | S.PGD-20 | S.CW | **S.AA** | NRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| 50ep | 75.81 | 1.100 | **64.28** | 26.12 | 26.73 | 24.38 | 35.35 |
| 100ep | 76.62 | 0.982 | 63.65 | 27.33 | 27.38 | 25.20 | 36.11 |
| 150ep | 77.52 | 0.887 | 62.93 | 28.06 | 27.57 | 25.78 | 36.51 |
| 200ep | 77.65 | 0.808 | 62.72 | 28.69 | 27.80 | 25.88 | **36.64** |
| 300ep | 78.32 | 0.712 | 62.24 | 28.81 | 27.59 | 25.80 | 36.48 |

**Every column is monotone in teacher training length, and the headline is a sign reversal:**

$$r(\text{teacher clean},\ \text{student clean}) = \mathbf{-0.999}.$$

The teacher gains $2.51$ points of clean accuracy across the ladder and the student **loses**
$2.04$. A more accurate teacher makes a worse student. What the student inherits is not the
teacher's accuracy.

NRR peaks in the middle ($200$ep, $36.64$), so there is an optimal teacher length rather than
"longer is better". The trade-off is a *dial*, and it costs nothing: the teacher is naturally
trained, the student config is untouched, and only the checkpoint path changes. **The
robust-teacher KD line does not have this knob** — there the teacher's geometry is fixed by
adversarial training.

Reproduced on Tiny-ImageNet with the same recipe and near-identical magnitude:

| | teacher clean $\Delta$ | student clean $\Delta$ | student AA $\Delta$ | NRR $\Delta$ |
|---|---:|---:|---:|---:|
| CIFAR-100, 50 → 200ep | +1.84 | **−1.56** | **+1.50** | +1.29 |
| Tiny-ImageNet, 80 → 200ep | +0.32 | **−1.92** | **+1.58** | +1.47 |

Tiny-ImageNet is the accuracy-controlled instance: its two teachers are $0.32$ apart
($65.97$ / $66.29$) and the student still moves by $1.9$ / $1.6$. Accuracy cannot be the cause there.

## 9.1 What actually changes in the teacher — not collapse, separation

$S_w/S_b$ falls monotonically ($1.100\to0.712$), which is the neural-collapse signature. But the
ratio hides which half moved, and splitting it changes the story. On the unit sphere:

| | 50ep | 300ep | $\Delta$ |
|---|---:|---:|---:|
| sample → its own class mean | 34.6° | 30.9° | −3.7° |
| class mean ↔ nearest class mean | 35.7° | **44.9°** | **+9.2°** |

Both move, but **separation moves $2.5\times$ more than concentration**. And on the student side
only one of them transfers:

| student | 50ep-teacher | 300ep-teacher | $\Delta$ |
|---|---:|---:|---:|
| sample → its own class mean | 27.9° | 28.2° | **+0.3° (flat)** |
| class mean ↔ nearest class mean | 25.4° | **29.8°** | **+4.4°** |

**The student inherits the class separation and not the within-class concentration.** So "the
teacher collapses" is the wrong description of the mechanism; **the teacher's class means spread
apart, and the student copies that.** $r(\text{teacher gap},\ \text{student gap}) = +0.999$.

## 9.2 Why one change moves both axes in opposite directions

**Clean falls because the student cannot follow.** The teacher's class means spread by $9.2°$
across the ladder; the student's spread by only $4.4°$ — **48% of it** — and the shortfall widens
(student/teacher gap ratio $0.711\to0.664$). So the further the teacher separates, the worse the
student reproduces it: $\cos(\hat\Phi_s,\hat\Phi_t)$ falls $0.8607\to0.8311$. Clean accuracy tracks
that fidelity, $r = \mathbf{+0.994}$.

The reason it cannot follow is the other half of its own objective: it has to hold the teacher's
clean value **while under attack**. Producing a large class separation and being locally constant
on $\varepsilon$-balls pull against each other, so the robustness requirement caps how much of the
teacher's separation the student can realize.

⚠ *An earlier version of this subsection attributed the clean drop to within-class "dark knowledge"
shrinking as the class scale grows. That was a guess. What the data shows is the simpler thing —
the student's fidelity to its own teacher falls — and it has the advantage of being the observable
this project already uses everywhere else: T.3's 17-cell correlation ($r=+0.83/+0.87$), §8.0's
three-objective ladder ($0.10/0.47/0.83$), and this ladder ($r=+0.994$) are all the same statement,
**clean accuracy is fidelity to the teacher**. The teacher ladder is a new handle on that one
quantity, not a second mechanism.*

**AA rises because the partial separation still outpaces the attack.** The student gains $+4.4°$ of
margin while the attack's rotation grows only $+1.9°$ ($16.8°\to18.7°$). Note the naive reading is
wrong: rotation and AA move *together* here, $r(\text{rotation},\text{AA}) = +0.83$. What tracks AA
is rotation priced against the margin it has to cross:

| teacher | attack rotation | class-mean gap | **rotation / (gap/2)** | AA |
|---|---:|---:|---:|---:|
| 50ep | 16.8° | 25.4° | **1.317** | 24.38 |
| 100ep | 17.2° | 26.8° | 1.286 | 25.20 |
| 150ep | 17.6° | 28.1° | 1.254 | 25.78 |
| 200ep | 18.1° | 28.8° | 1.259 | 25.88 |
| 300ep | 18.7° | 29.8° | 1.258 | 25.80 |

$r(\text{rotation}/\text{margin},\ \text{AA}) = \mathbf{-0.991}$, **and the saturation points agree** —
the ratio flattens at 150ep ($1.254/1.259/1.258$) and so does AA ($25.78/25.88/25.80$).

So one underlying change produces both halves:

$$\text{teacher class separation}\ \uparrow\ \Rightarrow\ \begin{cases}
\text{student falls further behind it} &\Rightarrow\ \text{fidelity}\ \downarrow\ \Rightarrow\ \text{clean}\ \downarrow\\
\text{the margin it does gain outpaces the attack} &\Rightarrow\ \text{AA}\ \uparrow
\end{cases}$$

## 9.3 Three readings that were tested and are wrong

1. **"A better teacher gives a better student."** Sign-reversed, $r=-0.999$. This is the finding, not
   a caveat.
2. **"Collapse means samples concentrating on their class mean."** On the student that quantity is
   flat ($27.9°\to28.2°$); the moving part is the separation between class means. §9.1.
3. **"Feature oscillation explains robustness."** Wrong sign here ($r=+0.83$), and this is the third
   time the project has found it: §8.7 has ADR and ours at matched oscillation with a 4.8 clean gap,
   and the teacher ladder has the more volatile teachers producing the more robust students. **Feature
   oscillation is not a sufficient statistic for robust accuracy** — the oscillation has to be priced
   against the margin it must cross. This qualifies §8.6, which leans on $O$ alone.

## 9.4 Limits

Single seed per cell; five teachers on CIFAR-100 and three on Tiny-ImageNet. The base regime only
(50 epochs, no WA/AWP) — the champion regime is untested on this axis. Teacher accuracy and teacher
separation move together on the CIFAR ladder, so CIFAR alone cannot separate them; Tiny-ImageNet is
what does, and it is three points. The margin used is the mean angle to the *nearest* other class
mean, which is a crude stand-in for the decision boundary. And no causal claim is made about *why*
longer natural training separates class means — that is a property of natural training this section
measures and does not explain.

**Framing warning.** This must not be written as "the teacher can be tuned per dataset" — the paper's
selling point is that CIFAR-10/100 run an identical config. The teacher is chosen once. What the
section claims is that **the teacher's geometry, not its accuracy, is what a feature anchor
transfers**, and that this gives a free dial the robust-teacher line does not have.
