# Method: Anchored Distillation from a Natural Teacher

*Shipped instantiation: a feature anchor with a sensitivity-matched per-sample attack radius.*

*Working draft, 2026-08-01; restructured 2026-08-07 so the surviving claims lead. Notation fixed
here is meant to carry into the paper.*

---

## Positioning

### P0. The ledger — which component moves which axis

Everything below is organized around one measured split. It is not a framing preference; it is what
the `nofeat` ablation (§4.1) and the sensitivity-matched-$\varepsilon$ stack (§8.0) force.

| component | axis | measured contribution | theory in this document |
|---|---|---|---|
| external, adversary-independent anchor at $x_{\mathrm{adv}}$ + two-block routing | **robust** | carries essentially all of the AA | §4 |
| schedule, WA, AWP, enlarged $\varepsilon_{\mathrm{tr}}$ | **robust** | $+0.82$ / $+0.56$ AA; off-the-shelf | **none** (§7.7) |
| $\mathcal{L}_{\mathrm{dir}}$ — the anchor in *feature* space, *magnitude-free* | **clean** | clean $+1.82$, CW $+0.50$, **AA $0$** | §5 |
| sensitivity-matched $\varepsilon$, $p=1$ | **clean** | clean $+1.4$–$2.1$, CW $+0.2$–$0.4$, **AA tie** | §3.2a, §5.7 |

Two readings follow, and the document commits to both.

**(a) The robustness is bought by the anchor, not by the direction loss.** Earlier drafts led with
row 3 as *the* mechanism. Deleting $\mathcal{L}_{\mathrm{dir}}$ outright leaves AA unchanged
($28.71$ vs $28.69$, §4.1), so that identification is retired. What survives — and is still not
standard — is the *shape*: an external target that does not depend on the adversary, imposed at
$x_{\mathrm{adv}}$, with the two TRADES terms routed to disjoint parameter blocks.

**(b) Everything this work adds beyond that shape buys clean accuracy at matched robustness.** That
is not a consolation prize; it is exactly the operating-point claim of P2. It also means the
mathematics of §5 is aimed at the axis it can explain rather than misdirected at the robust one —
answering the proportionality objection the previous draft had to concede (small lever, large
theory) by making the clean axis the headline instead of an afterthought.

### P1. The teacher is naturally trained — and this is a claim, not a fallback.

Robust knowledge distillation (ARD, RSLAD, ARREST, B-MTARD, …) requires an adversarially-trained
teacher, by far the most expensive precondition in that literature. We require only an ordinary
natural checkpoint. The argument of §2 and §4.2 is what licenses this: a natural teacher is precisely
a $(\mathcal{F}_t,w_t)$ pair whose feature set was selected by $\rho$ alone, and §4.2 Claim 1 computes
why anchoring the student to that teacher's *value* at $x_{\mathrm{adv}}$ selects the robust route
anyway. $\rho$ comes from the teacher, $\gamma$ comes from the inner maximization, and the two
sources reach disjoint parameter blocks by construction (Prop. 1) — so a *robust* teacher is not
merely unavailable here, it is **unnecessary**.

Two honest qualifications, both discharged later: the head is *initialized to* $w_t$ and supervised
by a target built from it, so "discard $w_t$" is shorthand for "the backbone never sees $w_t$"
(§3.1); and the comparison this claim really needs — same recipe, robust teacher — is an ablation we
have not run (§6), so P1 is currently supported by the *absence of a requirement*, not by a measured
penalty for meeting it.

### P2. The target is the clean/robust operating point, not AA alone.

Against the strongest baseline (ADR-full) the method is an **AA tie with a large clean gain**:
CIFAR-100 $+4.81$ clean at $+0.09$ AA, CIFAR-10 $+1.40$ clean at $+0.69$ AA (§8.0). Both AA deltas
sit inside this project's observed noise band, so the claim we make is *"recovers natural accuracy at
matched robustness"* — **not** an AA improvement. All comparisons therefore report clean accuracy
alongside every robust metric, and NRR is the headline scalar.

> **One decision rule for AA, applied everywhere in this document.** $|\Delta\mathrm{AA}|\le0.4$
> (the paired-seed noise floor of §6) is a **tie**, in our favour or against. Consequently: our
> $+0.09$ / $+0.69$ over ADR-full are ties, the earlier draft's "$+0.19$ beats ADR-full" (§8.1) is
> also a tie and has been relabelled, and the sensitivity-matched-$\varepsilon$ cell's $-0.10$ / $-0.02$ against our own
> $p{=}0$ run is a tie as well. No sentence in this document may call a sub-$0.4$ AA difference a
> win.

**Definitions of the derived metrics** (used throughout, defined here once): with $A_{\mathrm{nat}}$
clean accuracy and $A_{\mathrm{rob}}$ a robust accuracy,
$H=2A_{\mathrm{nat}}A_{\mathrm{rob}}/(A_{\mathrm{nat}}+A_{\mathrm{rob}})$ is their harmonic mean —
$H(\mathrm{pgd})$ and $H(\mathrm{cw})$ name the choice of $A_{\mathrm{rob}}$ — and
$\mathrm{NRR}=H$ evaluated at $A_{\mathrm{rob}}=\mathrm{AA}$, i.e. the natural/robust harmonic mean
against the strongest attack. NRR is the scalar we rank on.

### The thesis

The one-line thesis is *not* "make maximal use of what the clean teacher learned." It is the
sharper and partly opposite statement:

> A naturally-trained teacher's knowledge **cannot** be transferred as a whole. It splits into a
> transferable part (the feature set $\mathcal{F}_t$, carrying $\rho$) and a part that must be
> re-solved rather than copied — the combination $w_t$, which is $\gamma$-blind by construction.
> **The method is the mechanism that enforces that boundary**, by routing the two to disjoint
> parameter blocks. The design choices below (external anchor at $x_{\mathrm{adv}}$, stop-gradient,
> free head) follow from it.

*Scope note.* An earlier version of this thesis also claimed the **feature space** and the
**direction-only** form of the anchor as corollaries of it. The `nofeat` ablation (§4.1) shows the
anchor need not live in feature space to get the robustness, and §5 shows the direction-only form is
a clean-accuracy choice. Both remain part of the shipped method and both are defended — just on the
clean axis, not as the source of $\gamma$.

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

## 3. The method

### 3.1 Initialization (the "finetune" step)

Both networks start from the same naturally-trained checkpoint $\theta_t$:

$$\theta_s^{(0)}\;=\;\theta_t\quad\text{(backbone \emph{and} head)},\qquad \theta_t\ \text{frozen thereafter}.$$

This is not a warm start of convenience. By §2 the natural checkpoint is exactly a $(\mathcal{F}_t,w_t)$ pair whose $\mathcal{F}_t$ was selected by $\rho$ alone; the student begins holding both, and the training procedure below keeps $\mathcal{F}_t$ while re-solving $w$.

⚠ **"Discard $w_t$" is too strong, and the document uses it loosely elsewhere.** The student head is
*initialized to* $w_t$ and is then supervised by $z_t/\tau$, which is built from $W_t$ — so the head
is not free of the teacher's combination in any literal sense; §3.5 in fact makes the teacher's
cosine response the head target on purpose. The precise and defensible statement is the one
Prop. 1 proves: **the backbone never sees $w_t$.** $w$ is re-solved on the student's own feature
distribution (Claim 2) starting from the teacher's solution, under a target derived from it. Read
every later "discard $w_t$" as shorthand for that.

Note the two networks are not the same function at $t=0$: the teacher head sees $\Phi_t$, the student head sees $\sigma\hat\Phi_s$, so $z_s^{(0)}=z_t/\lVert\Phi_t\rVert$ up to the bias.

### 3.2 Inner maximization (attack)

$$x_{\mathrm{adv}}\;=\;\arg\max_{x'\in\mathcal{B}(x,\varepsilon_{\mathrm{tr}})}\ \big\lVert Q^\top\!\big(\hat\Phi_s(x')-\hat\Phi_t(x)\big)\big\rVert_2^2
\tag{1}$$

solved by $m$-step PGD with sign steps of size $\alpha$:

$$x^{(j+1)}=\Pi_{\mathcal{B}(x,\varepsilon_{\mathrm{tr}})}\Big(x^{(j)}+\alpha\,\mathrm{sign}\,\nabla_{x}\big\lVert Q^\top(\hat\Phi_s(x^{(j)})-\hat\Phi_t(x))\big\rVert^2\Big),\quad x^{(0)}=x+0.001\,\xi,\ \xi\sim\mathcal{N}(0,I).$$

Champion: $m=10$, $\alpha=2/255$, $\varepsilon_{\mathrm{tr}}=8.8/255$, $\varepsilon=8/255$ at evaluation.
**No label and no head — either network's — appears in (1).**

### 3.2a Sensitivity-matched $\varepsilon$ (`featdir_angeps_p`, 2026-08-04)

> ⚠ **Renamed 2026-08-31.** This was called "angular budget allocation", which is wrong twice over.
> The rule equalizes the first-order movement of the **loss**, not of an angle -- for a cosine target
> that is $\Delta\cos\theta$, and converting to $\Delta\theta$ carries a per-sample $1/\sin\theta$.
> And the shipped design is now the unnormalized $\ell_2$ anchor, which has no angle in it at all.
> The code knob keeps the identifier `featdir_angeps_p` and the configs keep their `*_angeps` names
> so that existing `results/` directories stay addressable; only the prose name changed.

Equation (1) maximizes a **feature angle**, but the constraint set $\mathcal{B}(x,\varepsilon_{\mathrm{tr}})$
is a **pixel** ball, and that is a mismatch. The same radius rotates different samples by wildly
different amounts, so a uniform $\varepsilon$ delivers a highly non-uniform *angular* attack — the
quantity the loss is actually written in. We therefore equalize the loss movement rather than the
pixel budget.

To first order the angle moved under an $\ell_\infty$ ball of radius $\varepsilon$ is
$\;\approx\varepsilon\cdot\lVert\nabla_x L_{\mathrm{dir}}(x)\rVert$, so with
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

⚠ **What the table asserts, and what the ablation permits.** Calling $\mathcal{L}_{\mathrm{dir}}$
"the boundary term" claims it is what supplies robustness. **The decisive ablation says it is not**
(§4.1, `nofeat` row): deleting $\mathcal{L}_{\mathrm{dir}}$ entirely and training the head alone on
$z_t/\tau$ at $x_{\mathrm{adv}}$ gives **AA 28.71 against the champion's 28.69**. So the boundary
slot is filled by *anchoring an external, adversary-independent target at $x_{\mathrm{adv}}$* — the
row that is genuinely load-bearing is the **anchor** and **space→parameters** routing, not the choice
of a feature-space direction over a logit-space distribution. $\mathcal{L}_{\mathrm{dir}}$'s
measured contribution is clean $+1.82$ / CW $+0.50$ / **AA $0$**. §4.7 (the smoothness reading)
survives this; the identification of $\mathcal{L}_{\mathrm{dir}}$ with $R_{\mathrm{bdy}}$ does not,
and the paper must not lead with it.

### 3.4 Three propositions

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
narrower and is the one §4.5 needs — **no term of the objective ever pushes $\lVert\Phi_s\rVert$ toward
$\lVert\Phi_t\rVert$**, since the head also sees only $\hat\Phi_s$. The student's feature magnitude is
left as a free internal quantity rather than being bound to an unreachable teacher target.

**Proposition 3 ($Q$ is a rotation at $k=d$).** *If $k=d$ then $Q\in O(d)$ and $\lVert Q^\top v\rVert=\lVert v\rVert$, so (1) and (2) reduce to the plain objectives*

$$\mathcal{L}_{\mathrm{dir}}=\mathbb{E}\big\lVert\hat\Phi_s(x_{\mathrm{adv}})-\hat\Phi_t(x)\big\rVert^2=\mathbb{E}\big[\,2-2\cos\angle\big(\Phi_s(x_{\mathrm{adv}}),\Phi_t(x)\big)\big].$$

The champion uses $k=d=512$, i.e. **no subspace restriction**: the projection is a no-op and $\mathcal{L}_{\mathrm{dir}}$ is exactly a cosine loss. ($k<d$ is retained in the code as an ablation knob; see §6.)

### 3.5 On the head target

$z_t=\lVert\Phi_t\rVert\,W_t\hat\Phi_t+b_t$, hence

$$\frac{z_t}{\tau}\;=\;\frac{\lVert\Phi_t\rVert}{\tau}\,W_t\hat\Phi_t+\frac{b_t}{\tau},
\qquad \frac{\mathbb{E}\lVert\Phi_t\rVert}{\tau}\approx\frac{13}{16}\approx0.81 .$$

At $\tau=16$ the soft target is therefore, up to a constant close to $1$, the teacher's **cosine** response — the temperature is not an arbitrary smoothing constant but the scale that converts the teacher's logits into a magnitude-free target consistent with $\mathcal{L}_{\mathrm{dir}}$.

### 3.6 Optional consistency term (off in the champion)

$$\mathcal{L}_{\mathrm{cons}}=\mathrm{KL}\Big(\mathrm{sm}\big(z_s(x)\big)\,\Big\Vert\,\mathrm{sm}\big(z_s(x_{\mathrm{adv}})\big)\Big),
\qquad \mathcal{L}\mathrel{+}= a(e)\,\lambda\,\mathcal{L}_{\mathrm{cons}},\quad a(e)=(e/E)^2 .$$

This is **literally the TRADES boundary term**. It is set to $\lambda=0$; see §6 for why.

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

## 4. Where the robustness comes from: an external anchor at $x_{\mathrm{adv}}$

This section answers *why the method is robust at all*, and it is ordered the way the evidence
arrived: the ablation first, the model second. §4.1 removes the component the earlier draft credited
and shows AA does not move; §4.2 computes what the surviving component buys; §4.3–§4.6 say what the
head must therefore do; §4.7 records the one property of $\mathcal{L}_{\mathrm{dir}}$ the ablation
leaves intact; §4.9 states what has no account here at all.

### 4.1 The decisive ablation — it is not the feature-space direction loss

The earlier frame's central structural claim — $\mathcal{L}_{\mathrm{dir}}$ *is* the boundary term,
hence the source of robustness — is **falsified** by deleting it:

| test | measurement | verdict |
|---|---|---|
| **delete $\mathcal{L}_{\mathrm{dir}}$ entirely**, keep everything else (head KD on $z_t/\tau$ at $x_{\mathrm{adv}}$, same schedule/WA/AWP/$\varepsilon_{\mathrm{tr}}$) — `nofeat_champ200_norm` | $58.92$ / CW $30.03$ / **AA 28.71** vs champion $60.74$ / $30.53$ / **AA 28.69** | ❌ **AA tie.** $\mathcal{L}_{\mathrm{dir}}$'s entire measured contribution is clean $+1.82$ / CW $+0.50$ / **AA $0$** |

Two consequences the paper has to absorb:

1. **Plain KD at $x_{\mathrm{adv}}$, with this schedule, already matches ADR-full on AA and leads it
   by $+1.56$ clean** (ADR-full $57.36$ / $28.50$; the AA $+0.21$ is a tie by §0).
   Whatever produces the headline robustness is the *anchoring-plus-schedule* combination, not the
   feature-space direction loss. The natural-teacher claim (P1) is untouched — it is if anything
   strengthened, since it now needs even less machinery — but the *directional* half of the story is
   demoted from mechanism to clean-accuracy lever.
2. Every row of the ablation table (§6) that reads "$\mathcal{L}_{\mathrm{dir}}$ occupies the
   boundary slot" is evidence about a term that can be removed for free on AA. They are not thereby
   wrong, but they cannot carry a robustness mechanism claim.
   *(`nofeat_champ200_raw`, the same ablation without normalization, diverges in the first epoch —
   AA $22.66$ — so the tie is specific to the normalized form.)*

### 4.2 What the anchor buys, computed exactly (Tsipras et al. model)

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

#### Claim 1 — matching the teacher's **raw** value at $x_{\mathrm{adv}}$ selects the robust route

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

**Why this survives §4.1.** The computation nowhere uses the fact that $\Phi_t$ is a *feature*. It
uses only that the anchor is (i) computed on the clean input by a frozen network and (ii) a proxy for
$y$ — properties the teacher's logit vector $z_t(x)$ has as fully as its penultimate feature. So the
model **predicts** the `nofeat` tie instead of being embarrassed by it: swapping the anchor from
$\hat\Phi_t(x)$ to $z_t(x)/\tau$ changes which route the argument is written about, not the argument.
What the model does *not* predict is the clean gap between the two anchors — that is §5's subject.

### 4.3 Claim 2 — the head, by contrast, **must** be re-solved

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

### 4.4 The angular form of Claim 2, and its quantitative prediction

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

### 4.5 The free head is the release valve of an unsatisfiable target

The target $\hat\Phi_t(x)$ is computed (i) on the **clean** input and (ii) by a **natural** teacher, while the constraint is imposed at $x_{\mathrm{adv}}$. Decompose the teacher's representation into $\gamma$-robust and useful-non-robust parts,

$$\Phi_t = \Phi_t^{(r)} + \Phi_t^{(n)} .$$

By definition $\Phi_t^{(n)}$ flips under some $\delta\in\mathcal{B}(0,\varepsilon)$; no student can reproduce it at $x_{\mathrm{adv}}$ for all $x$. Hence

> $\mathcal{L}_{\mathrm{dir}}$ **has a strictly positive floor** on the non-robust component: the objective demands that non-robust features be made robust.

This is a design property, not a defect, and the consequence the method acts on is the following.

**The head must stay free.** The residual the backbone *cannot* clear has to be discounted somewhere. The only place is $W_s$, which can down-weight the coordinates that failed to become robust. **The free head is the release valve of an unsatisfiable backbone target.** Freezing either its direction (gain-only head) or its magnitude (cosine head) closes the valve.

### 4.6 Claim 3 (corollary) — *how* the head is re-solved is secondary

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

### 4.7 $\mathcal{L}_{\mathrm{dir}}$ doubles as a smoothness term

The anchor is adversary-independent, so matching it at $x_{\mathrm{adv}}$ implicitly forces $\hat\Phi_s(x_{\mathrm{adv}})\approx\hat\Phi_s(x)$. This is why the loss is insensitive to the exact form of the inner attack, whereas KL-shaped losses require their matched adversary.

### 4.8 Caveats of the model (they apply to §4.2–§4.4 alike)


1. $\Phi$ is a scalar in the model, 512-dimensional for us; the multi-dimensional version needs
   $\mathbb{E}[ee^\top]$ (the residual covariance) rather than a single $e$.
2. $x_1$ is assumed unperturbable. Real images have no such clean separation.
3. The model's teacher *is* the bulk mean. A real natural network mixes robust and non-robust
   features, which shrinks Claim 1's $16\times$ margin by an unknown amount.
4. $\rho$ drifts during training; "the optimal head at fixed $\rho$" is an approximation.
5. In the model, $p=0.95$ caps robust accuracy — i.e. the ceiling is set by the **data**, not by the
   teacher. That is a usable implication rather than a limitation.

### 4.9 What has no theory here

The ledger's second row — one-cycle plus frozen LR, weight averaging, AWP, and
$\varepsilon_{\mathrm{tr}}=1.1\varepsilon$ — supplies $+0.82$ and $+0.56$ AA (§7.7) and is explained
by nothing in this document. Together with §4.1 this is the honest state of the mechanism account:
**the anchor shape is ours and is argued for; the magnitude of the robustness is largely bought by
standard machinery.** A reader who wants the paper's robustness claim to rest on new theory will not
find that here, and §8.0's stacked table is the place where the division is visible run by run.

---

## 5. Where the clean accuracy comes from: magnitude-free targeting

Per the ledger, this is the axis the method's own components actually move, and the axis P2 makes
the headline. The section is unchanged in substance from the previous draft's §2c; what changed is
that it is no longer competing with §4 to explain robustness.

§4.2 Claim 1 left one thing explicitly open: *"the model is scalar, so it cannot explain the $+0.95$
clean; that gap is stated, not resolved."* This section resolves it. The result is a clean division
of labor that **is** the trade-off resolution:

> **Robustness** is set by *what value* the backbone is anchored to (Claim 1, normalization-free).
> **Clean accuracy** is set by whether the student must also reproduce the anchor's magnitude
> (this section). Because the two live on different quantities, moving one need not cost the other —
> and §3 makes the split structural by routing them to disjoint parameter blocks.

This is why the empirical signature of removing normalization from **the backbone target** is
**clean $-0.95$ at robustness unchanged**, not a robustness move. The scalar model of §4.2 cannot see
this because it has no magnitude degree of freedom; the vector argument below supplies exactly that.

| cell | what is raw | clean | PGD-20 | CW | AA |
|---|---|---:|---:|---:|---:|
| champion ($p{=}0$) | nothing | 60.74 | 34.94 | 30.53 | 28.69 |
| `featdir_champ200_rawfeat` | direction loss only (head still reads $\hat\Phi_s$) | 59.79 | 34.94 | 30.13 | 28.57 |
| `featdir_champ200_fullraw` | direction loss **and** head input | 57.78 | **35.43** | 29.68 | 28.04 |

**Scope, stated up front.** The claim of this section is about the middle row — the *backbone
target*, holding the head input fixed at $\hat\Phi_s$. It is exactly there that PGD is unmoved to
three significant figures. The bottom row, which additionally un-normalizes the **head input**, is
outside the comparison defined below and does move robustness (PGD $+0.49$, AA $-0.53$ — in opposite
directions, so it is not a clean read either way). Normalization is a clean-accuracy lever *at the
backbone target*; the head-input normalization is a separate knob and this section does not license
a claim about it.

### 5.1 The two candidate losses, on a common decision rule

Write the backbone target as a value-match (Claim 1) and compare matching it with vs. without the
student's own magnitude as a free coordinate:

$$
\mathcal{L}_{L2}=\big\lVert \Phi_s(x_{\mathrm{adv}})-\Phi_t(x)\big\rVert_2^2,
\qquad
\mathcal{L}_{\cos}=\big\lVert \hat\Phi_s(x_{\mathrm{adv}})-\hat\Phi_t(x)\big\rVert_2^2 .
$$

Both are graded through the **same** head input $\hat\Phi_s$ (§0, $\sigma=1$), so the comparison is
of the *backbone target only*. The one property that separates them is invariance to the student's
feature magnitude, $\mathcal{L}_{\cos}(c\,\Phi_s,\Phi_t)=\mathcal{L}_{\cos}(\Phi_s,\Phi_t)\ \forall c>0$,
which $\mathcal{L}_{L2}$ lacks.

**Which side the invariance has to be on is *not* settled by our ablation, and the draft previously
overstated it.** The natural conjecture is that it is the *student* side that matters — only
$\hat\Phi_s$ makes the loss scale-free, whereas normalizing the target alone merely picks a different
constant to chase. But `featdir_champ200_rawfeat` does **not** test that: its config sets
`featdir_rawteacher: True` **and** `featdir_rawstudent: True`, i.e. it is the raw/raw cell, moving
both sides at once. The cells that separate the two sides (`n2_tnorm_sraw`, `n2_traw_snorm`) exist
only in the 50ep / no-WA / no-AWP screening regime, which §6's regime warning forbids citing about
the shipped method. **Status: conjecture, untested at the champion regime.** The prediction it makes
is sharp — a champion-regime `teacher_norm: True, featdir_rawstudent: True` cell should behave like
$\mathcal{L}_{L2}$ (clean down), not like $\mathcal{L}_{\cos}$ — and it is one run.

### 5.2 Assumption (capacity-limited residual)

The separation is not visible at infinite capacity — there both losses drive the residual to zero.
It appears under a **fixed representational budget**, where the student cannot perfectly fit its
target and the *unfittable* part is absorbed as residual noise on $\hat\Phi_s$. Model the fitted
student coordinate as $\hat z_{s,k}=\hat z_{t,k}+r_k$ with $r_k\sim\mathcal{N}(0,\sigma_\bullet^2/d)$
and posit

$$
\sigma_{\cos}^2 \;<\; \sigma_{L2}^2 .
$$

The mechanism is the gradient geometry: $\nabla_{\Phi_s}\mathcal{L}_{\cos}\propto
\tfrac{1}{\lVert\Phi_s\rVert}(I-\hat\Phi_s\hat\Phi_s^\top)(\hat\Phi_s-\hat\Phi_t)$, and the
projector $(I-\hat\Phi_s\hat\Phi_s^\top)$ **annihilates the radial component** — so $\mathcal{L}_{\cos}$
spends the entire (capacity-limited) budget on *angle*, the label-relevant coordinate. $\mathcal{L}_{L2}$
additionally forces the student to reproduce $\lVert\Phi_t\rVert$, which **in this model** carries no
class information ($\mathbb{E}[\lVert\Phi_t\rVert^2\mid y]=d(1+\eta^2)$, free of $y$) — a nuisance
that competes with angular alignment for the same budget, inflating $\sigma_{L2}^2$.
*(This is an assumption about optimization under a capacity constraint the toy model does not itself
impose; we flag it as such and let the ablation adjudicate — see below.)*

⚠ **The $y$-independence of $\lVert\Phi_t\rVert$ is an artifact of the toy model, and the rest of
this document relies on the opposite.** Prop. 2 reading 2 treats $\lVert\Phi_s\rVert^{-1}$ as a
*useful* implicit per-sample weight ("large-norm = easy/confident"), which presumes the norm carries
difficulty information; the same presumption sits behind the norm-based temperature line of work in
this project. Both readings cannot be load-bearing at once. The reconciliation we actually believe:
the norm carries **difficulty** information but not **class** information, so it is a nuisance for
the *class* decision (this section) while still being a usable *weighting* signal (Prop. 2). That
distinction is asserted here, not measured — and the measurement that would settle it (norm
dispersion across classes vs across difficulty) came back showing dispersion on CIFAR is small
enough that Prop. 2's re-weighting is close to inert, which weakens Prop. 2's side of it rather
than this one's.

### 5.3 Consequence: strictly higher clean SNR

With the head reading $\hat\Phi_s$ and the uniform read-out $w=\mathbf 1/\sqrt d$, the class signal at
the decision rule is $w^\top\hat\Phi_s$. Write the two averages that control it: by the SLLN,
$\tfrac1d\sum_k z_{s,k}\to\eta y$ and $\tfrac1d\lVert\Phi_s\rVert^2\to1+\eta^2+\sigma_\bullet^2$.
**At finite $d$** (the limit itself is degenerate — see the caveat), Slutsky gives the standard
Gaussian approximation

$$
w^\top\hat\Phi_s\ \approx\ \mathcal{N}\!\Big(\ \sqrt d\;\underbrace{\frac{\eta\,y}{\sqrt{1+\eta^2+\sigma_\bullet^2}}}_{=:\ \mathrm{snr}(\sigma_\bullet^2)}\ ,\ 1\Big),
\qquad
R_{\mathrm{nat}}\;\approx\;\Phi\big(-\sqrt d\;\mathrm{snr}(\sigma_\bullet^2)\big).
$$

The $\sqrt d$ is not cosmetic: it is what keeps $R_{\mathrm{nat}}$ a non-degenerate probability. In
the strict $d\to\infty$ limit the decision statistic concentrates on a constant and the error goes to
$0$ or $1$, so the $\Phi(\cdot)$ expression is a finite-$d$ statement throughout — the previous draft
took the limit and then kept the CDF, which is inconsistent.

$\mathrm{snr}$ is strictly decreasing in $\sigma_\bullet^2$
($\partial_{\sigma^2}\mathrm{snr}=-\tfrac12\eta(1+\eta^2+\sigma^2)^{-3/2}<0$), so
$\mathrm{snr}(\sigma_{\cos}^2)>\mathrm{snr}(\sigma_{L2}^2)$ and, $\Phi$ being increasing,
$R_{\mathrm{nat}}$ is strictly smaller under $\mathcal{L}_{\cos}$. (The model is binary, $y=\pm1$;
there is no per-class sum and no $K$ here — the earlier $\tfrac1K\sum_c$ was imported from the
$K$-class setting by mistake, and the bias $b$ only shifts the threshold, it does not enter the
comparison since both losses share it.)

**This is a clean-accuracy statement only.** It does *not* claim a robustness gain — consistent with
§4.2 Claim 1, which already sources robustness from value-matching, and with the rawfeat tie.

### 5.4 The assumption, measured (2026-08-06)

The argument is entirely about the angular residual, and that residual is directly observable: $\cos$
is defined identically whether the loss is raw or normalized, so the two runs are comparable on it.
Measured on the **full CIFAR-100 test set** ($n=10000$), both `last` checkpoints, teacher
`clean_200ep`:

| run | backbone target | $\cos_{\text{clean}}$ | $\cos_{\text{adv}}$ (own attack) | $\cos_{\text{adv}}$ (matched attack) | $\lVert\Phi_s\rVert$ |
|---|---|---:|---:|---:|---:|
| champion | $\mathcal{L}_{\cos}$ | **0.8245** | 0.7243 | 0.7243 | 34.42 |
| `champ200_rawfeat` | $\mathcal{L}_{L2}$ | 0.8178 | 0.7257 | 0.7242 | 8.71 |
| teacher | — | — | — | — | 11.20 |

*("Matched attack" = both students attacked with the champion's normalized-direction adversary, so
the column is not confounded by the two runs training against different inner problems.)*

**Verdict: the prediction holds on clean inputs and fails at $x_{\mathrm{adv}}$.**

1. $\cos_{\text{clean}}$ moves in the predicted direction, $+0.0067$ for $\mathcal{L}_{\cos}$. The
   implied change in the decision statistic is $+0.8\,\%$ relative, against a measured clean gain of
   $+1.6\,\%$ relative ($59.79\to60.74$) — the same order of magnitude, which is the most this
   argument can honestly claim.
2. **At the adversarial point the two are identical** ($0.7243$ vs $0.7242$ under the matched
   adversary). But $x_{\mathrm{adv}}$ is exactly where $\mathcal{L}_{\mathrm{dir}}$ is evaluated, so
   the literal statement $\sigma_{\cos}^2<\sigma_{L2}^2$ *at the point the loss acts on* is **not
   supported**: both targets fit their own objective equally well in angle. What differs is only how
   that alignment **generalizes to clean inputs**. The capacity-competition story told above predicts
   a gap at $x_{\mathrm{adv}}$ and there is none; it should be replaced by, or at least narrowed to,
   a generalization statement.
3. Prop. 2 is confirmed as a side effect: the cosine-trained student's feature norm floats free to
   $34.4$ ($3\times$ the teacher), while the $L2$-trained one is pulled to $8.7$, near the teacher's
   $11.2$ — the raw target does bind the magnitude, the normalized one does not.

Reproduce with `scratchpad/measure_cosadv.py` (no training; two forward passes plus the 10-step inner
attack over the test set).

> **Why this does not repeat the robust half of the older draft.** An earlier version of this
> argument (in `direction_guided_at_section.md`) ran the same SNR inequality through the robust
> error as well, predicting $\mathcal{L}_{\cos}$ dominates on *both* axes. The rawfeat ablation
> falsifies the robust half (PGD/AA unchanged), so we keep only the natural-error conclusion. The
> robust axis is governed by Claim 1, which is normalization-free by construction; the SNR account
> is retired to exactly the axis the data supports.

### 5.5 Same mechanism, second signal: soft-label distillation

The head target is $z_t/\tau$, not the hard label $y$ (§3.5). The same variance-removal logic
applies to *its* supervision. Under a near-Bayes teacher $p_t(x)=\Pr(y{=}{+}1\mid x)$, a hard label
is a Bernoulli draw with $\mathrm{Var}(y\mid x)=4p_t(1-p_t)$ — **largest exactly at the boundary**,
the samples robustness depends on — whereas $z_t/\tau$ is deterministic given the teacher. By the
usual least-squares variance $\mathrm{Var}(\hat W_s)\propto\mathrm{Var}(\text{target}\mid x)/n$, the
soft target injects less estimation noise into the head.

**Two things keep this from being an argument that KD beats CE.**

*First, it is a variance argument with the bias term omitted.* KD trades variance for bias toward the
teacher's decision, and our teacher is **natural**, i.e. $\gamma$-blind by P1's own construction. The
region where the variance saving is largest — $p_t\approx\tfrac12$, the boundary — is exactly the
region where a natural teacher's soft label is confidently uninformative about robustness. So the two
effects are largest in the same place and pull in opposite directions; the net sign is not decided
here.

*Second, the measurement says the net is a tie.* §4.6 Claim 3 compares CE and KD heads directly and
finds $+0.15$ / $+0.40$ CW — inside this project's $\pm0.3$–$0.4$ noise floor. **Claim 3 is the
binding evidence and this subsection must not be read as overturning it.** The correct reading is the
weaker one: the SNR account explains why softening the head target is *not harmful* and gives an
upper bound on how much it could have bought, and Claim 3 shows that in practice it buys nothing
measurable. The head target is $z_t/\tau$ for the reason in §3.5 (it is the magnitude-free target
consistent with $\mathcal{L}_{\mathrm{dir}}$), not because it wins on SNR.

With that caveat, the two levers can be *summarized* — not derived — as acting on one effective
signal-to-noise ratio,

$$
\mathrm{snr}_{\mathrm{tot}}\;\sim\;\frac{\eta}{\sqrt{1+\eta^2+\underbrace{\sigma_\bullet^2}_{\text{backbone target}}+\ \gamma\underbrace{\mathrm{Var}(y\mid x)}_{\text{head target}}}},
$$

which is a **mnemonic, not an equation**: $\sigma_\bullet^2$ is a residual variance on the feature
while $\gamma\,\mathrm{Var}(y\mid x)$ is an estimation variance on $\hat W_s$, and the constant
$\gamma$ that would make them commensurable is not derived anywhere. What survives literally is the
qualitative statement: feature-normalization and label-softening are the same
nuisance-variance-removal move applied to the two disjoint blocks — the backbone target and the head
target respectively — which is the SNR reading of why §3's split is natural rather than ad hoc.

### 5.6 Caveats specific to §5

1. The residual-variance gap $\sigma_{\cos}^2<\sigma_{L2}^2$ was **assumed**, motivated by the
   gradient projector but not derived, and it is close to a restatement of the conclusion. The
   measurement above is the only non-circular adjudication of it, and it came back **split**: the gap
   exists on clean inputs ($+0.0067$) and is **absent at $x_{\mathrm{adv}}$** ($+0.0001$), which is
   where the loss actually acts. The capacity-competition mechanism as written predicts the opposite
   pattern. Until that is repaired, §5 should be read as *"the normalized target generalizes its
   angular alignment to clean inputs slightly better,"* which is an observation, not a derivation.
2. The SNR statement is scalar-signal and finite-$d$; the $d\to\infty$ limit is degenerate and must
   not be used with the $\Phi(\cdot)$ error expression. A genuine finite-$d$ treatment needs the
   residual covariance $\mathbb{E}[rr^\top]$, as in §4.8 caveat 1.
3. The account is silent on the observed $+0.95$ clean *magnitude* — it predicts the **sign and the
   axis** (clean, not robust), which is what §4.2 left open, not the size.
4. Which side of the loss the scale-invariance must sit on (student vs teacher) is **conjecture**;
   the cell usually cited for it (`featdir_champ200_rawfeat`) moves both sides at once.
5. The soft-label subsection is a variance-only argument whose net effect is measured as a tie
   (§4.6 Claim 3); it does not support a KD-over-CE claim.
6. **Proportionality.** This section theorizes a $+0.95$ clean effect inside a component whose total
   contribution is clean $+1.82$ / CW $+0.50$ / **AA $0$** (§4.1, `nofeat` row). The levers that supply
   most of the headline AA — WA, AWP, the enlarged $\varepsilon_{\mathrm{tr}}$, the schedule — have
   no theory here at all (§4.9, §7.7). A reader is entitled to notice that the small lever is the one that
   got the mathematics.

### 5.7 The second clean lever: sensitivity-matched $\varepsilon$

The sensitivity-matched $\varepsilon$ of §3.2a is the other component in the ledger's clean rows, and it has the same
signature by a different route. It does not touch the target at all — it reallocates a *fixed total*
attack budget across samples so that the inner problem is equally hard in the geometry the loss is
written in (§3.2a property 1). Measured on both datasets it moves clean $+1.4$ to $+2.1$ and CW
$+0.2$ to $+0.4$ at an AA tie (§8.0), and it is the only intervention so far to transfer from the
50ep screening regime to the champion recipe.

There is no SNR-style account of it here. The working empirical rule it obeys — *touch the target and
clean moves; touch the attack and robustness moves* (§8.0) — is stated as a regularity, and the
the radius rule is the one intervention that touches the attack and yet lands on clean, which the rule
does not explain. That is an open item, not a result.

---

## 6. Predictions, and how the ablations land

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

**Read every row below through §4.1.** The `nofeat` ablation was promoted out of this table into
§4.1 because it governs how the table is read: every row phrased as
"$\mathcal{L}_{\mathrm{dir}}$ occupies the boundary slot" is evidence about a term that can be
deleted for free on AA. Those rows are not thereby wrong, but none of them can carry a
robustness-mechanism claim.

The remaining predictions, with that caveat in force:

| prediction (from §2/§4) | test | measurement | verdict |
|---|---|---|---|
| $w$ must be re-solved ⇒ freezing **either** channel of the head hurts, **symmetrically** | head $2\times2$ | free/free $H=41.77$; cosine head $40.01$; gain-only head $39.99$ | ⏳ symmetric $-1.8$ **but stale regime — no valid evidence at the shipped regime** |
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

Predicted: gap $-1.8\to\approx0$. No interaction ⇒ the causal story in §4.5 is wrong.

*This is a diagnostic, not a change to the method* — the shipped method uses a natural teacher (P1).
The robust teacher is an instrument for isolating the mechanism and appears in the paper as two
ablation rows. It requires no new training: `CIFAR100/checkpoint/at_ce_freehead/madry_at_last.pkl`
(clean-init AT, $58.45/27.23$; shares the coordinate basis, so it is usable as both teacher and
initialization) and `CIFAR100/checkpoint/at_teacher/madry_at_last.pkl` (scratch AT) are on disk.
Prefer the clean-init one: a scratch-AT model lives in a different basis, which invalidated the
earlier robust-PCA cell for the same reason.

---

## 7. Honest limitations

1. **The TRADES guarantee does not transfer.** Thm. 3.1 bounds $R_{\mathrm{rob}}-R^*_{\mathrm{nat}}$ when *both* terms use the same $f$. Replacing the boundary anchor with an external frozen teacher voids the bound. Our objective is TRADES-**shaped**, not TRADES-**justified**.
2. **We use only Ilyas §2** (the $C=(\mathcal{F},w,b)$ decomposition and $\rho/\gamma$). Their Theorems 1–3 are binary-Gaussian/linear; an earlier attempt to route the $k$-story through the $\Sigma\to I$ argument was falsified by the $k=350$ vs $k=512$ ablation. Do not re-import it.
3. **$w_c$ is per-class, Ilyas's $w$ is per-feature.** The mapping (row $w_c$ = weighting of the $d$ features for class $c$) is natural but must be stated, not assumed.
4. **The head target is a soft teacher distribution, not hard CE**, so its correspondence with $\psi^{-1}(R_\phi-R^*_\phi)$ in Thm. 3.1 is loose.
5. **Feature-space (rather than logit-space) boundary term** is **not** justified by either source paper, and no longer justified by reachability either (Appendix A.1); §4.1 shows it is not required for the robustness at all.
6. **Single seed** on essentially every cell above.
7. **WA, AWP and the enlarged $\varepsilon_{\mathrm{tr}}$ are off-the-shelf** and account for a large share of the final number ($+0.82$ and $+0.56$ AA respectively).

---

## 8. Current numbers

### 8.0 Champion — sensitivity-matched $\varepsilon$ on both datasets (2026-08-04)

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
schedules (the base row is 50ep, the $+$WA rows are 100ep). Both are listed in §7.

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

### 8.1 The preceding champion — ADR matched on AA, led on clean (2026-08-01)

| model | clean | PGD-20 | CW | AA | H(pgd) | H(cw) | NRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ours, `featdir_champ200_100ep`** | **60.74** | **34.94** | **30.53** | **28.69** | **44.36** | **40.64** | **38.97** |
| ADR-full (strongest baseline) | 57.36 | 34.92 | 30.62 | 28.50 | 43.41 | 39.93 | 38.08 |
| other-server champion (target) | 60.67 | — | — | 28.42 | — | — | 38.71 |
| our previous best (100ep+AWP, $\varepsilon_{\mathrm{tr}}$10) | 60.04 | 34.36 | 29.31 | 27.36 | 43.71 | 39.39 | 37.59 |
| 50ep champion | 62.75 | 33.96 | 28.41 | 26.29 | 44.07 | 39.11 | 37.06 |

**This is the first configuration that matches ADR-full on AA while leading clean by +3.38.** By the
decision rule of §0 the AA difference is a **tie**, not a win: $+0.19$ sits inside the $\pm0.4$ noise
floor, as do PGD-20 $+0.02$ and CW $-0.09$. What the row actually establishes is the P2 signature —
*matched robustness, recovered natural accuracy*. Every derived metric is a project record: H(pgd)
44.36, H(cw) 40.64, NRR 38.97 (vs ADR 38.08); NRR is the scalar that carries the improvement, and it
does so through the clean axis. Against the other-server champion it was reproducing, AA
$28.42\to28.69$ is likewise a tie at $+0.27$.

The P1/P2 reading: **robustness matched to the strongest baseline and clean accuracy well above it,
obtained without an adversarially-trained teacher.** The claim is a cheaper teacher at equal
robustness with a large clean gain — *not* a strict improvement on both axes.

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

## Appendix A. Retired arguments (kept for the record)

These were load-bearing in the 2026-08-01 draft and are retained so the paper does not silently
re-derive them.

### A.1 Unreachability as the justification for normalization

The original argument justified the magnitude-free target by *unreachability*: $\lVert\Phi_t\rVert$
is largely built from $\Phi_t^{(n)}$ (§4.5), so matching it imposes a target no student can meet at
$x_{\mathrm{adv}}$, leaving a residual gradient absorbed by memorization.

~~**Magnitude must be dropped.** $\lVert\Phi_t\rVert$ is largely built from $\Phi_t^{(n)}$ (teacher confidence tracks the non-robust bulk), so matching magnitude imposes an unreachable target, leaving a residual gradient that is absorbed by memorization. The unit direction is the reachable part of the request; Prop. 2 makes the neutrality exact.~~ **Retired** — `rawfeat` keeps the magnitude and loses no robustness. Prop. 2 remains true as stated (and is now measured: §5.4, $\lVert\Phi_s\rVert=34.4$ under $\mathcal{L}_{\cos}$ vs $8.7$ under $\mathcal{L}_{L2}$, teacher $11.2$); what is retired is the *inference from unreachability to necessity*.

§4.2 Claim 1 gets the robust prediction without unreachability, and §5 handles the clean axis. Note
that §4.5 — the free head as release valve — is *not* retired with it: that argument needs only the
positive floor of $\mathcal{L}_{\mathrm{dir}}$, not the inference about magnitude.

### A.2 The robust half of the SNR account

An earlier version (in `direction_guided_at_section.md`) ran §5's SNR inequality through the robust
error as well, predicting $\mathcal{L}_{\cos}$ dominates on *both* axes. The `rawfeat` ablation
falsifies the robust half, so only the natural-error conclusion is kept; see the box at the end
of §5.

### A.3 $\mathcal{L}_{\mathrm{dir}}$ as the boundary term of TRADES

Retired by §4.1. It survives in §3.3's comparison table only as a *shape* correspondence, with the
warning attached there, and the paper must not lead with it.
