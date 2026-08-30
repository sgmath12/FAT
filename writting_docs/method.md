# Method

Draft, 2026-08-30. Structure follows `reference/IGDM.pdf` §3 (an observation that motivates, the
derivation, then the module and its integration) and `reference/ADR.pdf` §3 (preliminaries, the
proposed target, the algorithm box, implementation). Companion: `introduction.md`.

---

## 3. Method

### 3.1 Preliminaries

Let $f = h \circ \Phi$ be a classifier decomposed into a feature map $\Phi : \mathcal{X} \to
\mathbb{R}^d$ and a linear head $h$. Adversarial training (Madry et al., 2018) solves

$$\min_\theta\ \mathbb{E}_{(x,y)}\ \max_{x' \in B(x,\varepsilon)}\ \ell\big(f_\theta(x'),\,y\big),
\qquad B(x,\varepsilon) = \{x' : \lVert x'-x\rVert_\infty \le \varepsilon\},$$

with the inner maximization approximated by projected gradient descent. The supervisory signal is the
label $y$, and it is asserted uniformly over $B(x,\varepsilon)$: every point of the ball is required
to attain full confidence in one class. This is the source of the accuracy cost. Adversarial
distillation replaces $y$ with a teacher quantity, and existing methods take that quantity from the
teacher's logits — at the clean input (ARD, RSLAD), or at the adversarial input (AdaAD), or through
finite differences of both (IGDM) — with a teacher that has itself been adversarially trained.

We keep the inner maximization exactly as in standard adversarial training, so that all robustness
continues to be produced by the student, and change only what the outer minimization is asked to
preserve.

### 3.2 Clean feature anchoring

Let $\Phi_t$ be the feature map of a network of the same architecture trained naturally on the same
data, frozen. The student's feature map $\Phi_s$ is initialized at $\Phi_t$ and trained with

$$\boxed{\ \mathcal{L}_{\mathrm{anchor}} \;=\; \mathbb{E}_{x}\,
\big\lVert \Phi_s(x_{\mathrm{adv}}) - \Phi_t(x) \big\rVert^2 \ },\qquad
x_{\mathrm{adv}} \;=\; \arg\max_{x' \in B(x,\varepsilon)}\ \big\lVert \Phi_s(x') - \Phi_t(x)
\big\rVert .$$

The attack maximizes the same quantity the outer problem minimizes, so the inner and outer problems
are matched, as they are in standard adversarial training and unlike methods whose attack is
generated against a different objective from the one being trained.

Three properties follow from the form alone.

**The teacher is evaluated at $x$ and nowhere else.** $\Phi_t$ never receives a perturbed input, in
the attack or in the loss. Whatever the teacher does inside $B(x,\varepsilon)$ is therefore outside
the objective and cannot be inherited — which is the entire answer to the objection that a naturally
trained teacher would transfer its fragility. The same fact means the teacher cannot contribute
robustness either: it supplies a target, and the work of holding that target under perturbation is
done by the student's own inner maximization.

**The objective is exactly zero at initialization.** The student begins as the teacher, so
$\Phi_s(x) = \Phi_t(x)$ and the loss is nonzero only because of the perturbation. Every gradient the
backbone receives is therefore attributable to the attack, which is the quantity to be trained. A
logit target does not have this property except at temperature $1$, where it carries no dark
knowledge.

**There is no weight and no temperature.** The anchor is the whole backbone objective rather than a
regularizer added to cross-entropy, so no coefficient balances it against anything. This also makes
the decomposition in §3.3 available: it characterizes the objective only when the anchor *is* the
objective.

### 3.3 What a single anchor term controls

The anchor looks like a fidelity constraint, and a fidelity constraint alone would have no reason to
produce robustness. It is not one. Fix $x$, write $B = B(x,\varepsilon)$, and define

$$L=\max_{x'\in B}\lVert\Phi_s(x')-\Phi_t(x)\rVert,\qquad
F=\lVert\Phi_s(x)-\Phi_t(x)\rVert,\qquad
O=\max_{x',x''\in B}\lVert\Phi_s(x')-\Phi_s(x'')\rVert,$$

so that $L$ is the quantity the method minimizes, $F$ is fidelity to the teacher at the clean point,
and $O$ is the oscillation of the student's own feature map over the ball.

> **Proposition 1.** $\quad L \;\le\; F + O \;\le\; 3L.$
>
> *Proof.* $F \le L$ because $x \in B$. $O \le 2L$ by inserting $\Phi_t(x)$ between $\Phi_s(x')$ and
> $\Phi_s(x'')$ and applying the triangle inequality to each half. And for any $x' \in B$,
> $\lVert\Phi_s(x')-\Phi_t(x)\rVert \le \lVert\Phi_s(x')-\Phi_s(x)\rVert + F \le O + F$; taking the
> maximum over $x'$ gives $L \le F+O$. $\square$

Minimizing the single term therefore controls teacher fidelity and local stability jointly, to within
a factor of three, with no coefficient trading them off. The second is what standard adversarial
training obtains from the label term and the first is what it gives up; the anchor obtains both from
one quantity, and it does so without ever querying the teacher off the clean point.

The prediction is verifiable and we verify it: on trained checkpoints, over the same
$\varepsilon$-balls, the student is markedly more stable than the teacher it was distilled from.

| | $L$ | $F$ | $O_{\text{lb}}$ | teacher's own $O$ |
|---|---:|---:|---:|---:|
| raw target | 7.583 | 6.304 | 2.164 | **6.684** |
| normalized target | 0.674 | 0.542 | 0.238 | **0.566** |

The student's oscillation is $2.4\times$ smaller than the teacher's, matching an independent angular
measurement ($14.6^\circ$ against $63.8^\circ$ of feature rotation under $\varepsilon = 8/255$), and
most of what remains in the loss is fidelity rather than instability.

**What Proposition 1 does not establish.** It bounds $F+O$, and $O$ is not a sufficient statistic for
robust accuracy: across the teacher ladder of §4 it correlates with AutoAttack at $r = +0.83$, the
wrong sign. The proposition explains why anchoring to a non-robust teacher is *free*; it is not a
robustness bound and we do not present it as one.

### 3.4 Sensitivity-matched $\varepsilon$

A fixed pixel radius does not correspond to a fixed amount of pressure on the objective. Within an
$\ell_\infty$ ball of radius $e$, the first-order change of any loss is

$$\max_{\lVert\delta\rVert_\infty \le e} \langle \nabla_x \mathcal{L},\,\delta\rangle
\;=\; e\,\lVert \nabla_x \mathcal{L} \rVert_1 ,$$

so with $g_i = \lVert\nabla_x\mathcal{L}(x_i)\rVert$ the same radius moves the objective by amounts
that differ across samples by whatever $g_i$ varies by. Equalizing that movement subject to a fixed
total budget $\sum_i \varepsilon_i = N\varepsilon$ gives

$$\varepsilon_i \;\propto\; g_i^{-p},\qquad
\varepsilon_i \leftarrow \mathrm{clip}\big(\varepsilon_i,\,0.5\varepsilon,\,1.5\varepsilon\big),\qquad
\varepsilon_i \leftarrow \varepsilon_i \cdot \frac{N\varepsilon}{\sum_j \varepsilon_j},$$

with $p = 1$ throughout, the clip preventing degenerate radii on near-flat samples, and the final
rescaling restoring the mean exactly.

**Budget preservation is what makes the comparison interpretable.** $p{=}1$ and $p{=}0$ spend the
same total attack budget over the dataset, so any difference between them is allocation and not
strength — the rule cannot be dismissed as attacking harder. Measured with no other component
present (no weight averaging, no AWP), it is worth $+1.77$ standard accuracy at $+0.19$ AutoAttack
over 100 epochs and $+1.61$ at $+0.21$ over 50, i.e. **the same $+0.49$ NRR at either schedule
length**.

*Qualifications, stated rather than buried.* The rule equalizes the movement of the **loss**, not of
an angle; an earlier name for it, "angular budget", was wrong for that reason and the code knob
retains the old identifier. $g_i$ is one backward pass at the clean $x_i$, i.e. the sensitivity at
the starting point of the 10-step attack rather than along it. The derivation asks for the $\ell_1$
norm and our runs used $\ell_2$; both were trained and agree within noise (62.51 / 28.44 against
62.35 / 28.68).

### 3.5 The complete method

**Algorithm 1** — Clean Feature Anchoring (CFA)

```
Require: dataset D, radius eps, steps K, step size a, exponent p = 1
 1: Phi_t <- train f_t naturally on D                       # same architecture, no adversarial cost
 2: Phi_s <- Phi_t                                          # student initialized at the teacher
 3: for each epoch do
 4:     for each minibatch {x_i} in D do
 5:         g_i <- || grad_x || Phi_s(x_i) - Phi_t(x_i) ||^2 ||            # one backward pass at x
 6:         eps_i <- clip(eps * (g_i / mean g)^(-p), 0.5 eps, 1.5 eps)
 7:         eps_i <- eps_i * N eps / sum_j eps_j                           # restore total budget
 8:         x_i^adv <- PGD_K( max_{x' in B(x_i, eps_i)} || Phi_s(x') - Phi_t(x_i) || )
 9:         L <- mean_i || Phi_s(x_i^adv) - Phi_t(x_i) ||^2
10:         update Phi_s by one optimizer step on L
11:     end for
12: end for
```

Line 1 is the only place labels are used, and it is ordinary supervised training. Lines 3–12 use no
labels at all.

**What the classifier does.** The head is inherited from the teacher and is never part of the
backbone objective: the anchor is computed on features, the attack is computed on features, and the
head appears in neither. On Tiny-ImageNet the head is frozen outright. On CIFAR-10 and CIFAR-100 the
runs additionally carry a *detached* refinement of the head toward the teacher's softened logits
($\tau = 16$), inherited from an earlier version of the method; because it is detached it cannot
reach the backbone, and the ablation in §7 shows it is removable at no cost — leaving the head
exactly as the teacher's scores $+0.29$ NRR above training it. We report it because it is present in
the runs, not because it is needed.

**Implementation.** ResNet-18 throughout. The teacher is the same architecture trained naturally for
200 epochs on all three datasets; §4 varies this deliberately, and it is the only knob we vary per
dataset. The student trains for 100 epochs
with AdamW at learning rate $0.021$ under a one-cycle schedule, batch size 128, with a 10-step attack
at step size $2/255$ and training radius $8.8/255$; evaluation uses $\varepsilon = 8/255$. Weight
averaging ($\kappa = 0.999$, from 20% of the run) and an AWP proxy ($\gamma = 0.005$, after epoch 10)
are standard machinery that our baselines also carry, and §7 reports the method with and without
them. **Every dataset uses this identical configuration; only the dataset name and the teacher
checkpoint change.** Evaluation is AutoAttack (standard version) on the full test set, alongside
PGD-20 and CW.

---

## Writer's notes

### One discrepancy this draft resolves, and the decision it needs

`writing.md` §2 currently states that "the classifier stays at the teacher's and is never trained".
That is true of the Tiny-ImageNet runs (`featdir_freeze_head: True`) and it is what the ablation
recommends, but it is **not** true of the CIFAR-10 and CIFAR-100 runs as executed: those configs set
`beta: 1.0`, `tau: 16` and omit `featdir_freeze_head`, so the head is fitted by
$\beta\,\mathrm{KL}(h(\Phi_s(x_{\mathrm{adv}}))\,\Vert\,z_t/\tau)$. The term is detached from the
backbone (`featdir_alpha` defaults to $0$, `methods.py:2106`), so it changes no backbone gradient and
no adversarial example, but it does change the final classifier and therefore the reported numbers.
§3.5 above states this as it is.

**Resolved 2026-08-31: option 1, re-run with the head frozen.** `l2_bestrecipe_freezehead` is
`l2_bestrecipe_angeps` plus `featdir_freeze_head: True` and nothing else, launched 02:01. The
base-regime ablation says the frozen head is better (NRR 36.64 against 36.35), so this is expected to
hold or improve 62.35 / 28.68 while making the description exact. If it does, $\tau$ and $\beta$
leave the method entirely and "no weight and no temperature" becomes literally true rather than
scoped to the backbone; §3.5's paragraph on the classifier then reduces to one sentence and the
Tiny-ImageNet configuration becomes the description of all datasets rather than the exception.

No `feat_scale` is involved. That knob exists because a *directional* student hands the frozen head a
unit-norm vector against a bias calibrated for norm $\approx 11.2$; this cell is `student_norm:
False`, so `head_hat = bool(student_norm)` is false, `scale` stays at $1.0$, and `head_from_feat`
receives the student's own raw feature unmodified.

### Which design the method section describes

§3.2 states the objective as an unnormalized $\ell_2$ anchor, matching `writing.md` §2. The runs
backing that are `l2_bestrecipe_angeps` on CIFAR-100 (62.35 / 28.68) and `featdir_tin_100ep` on
Tiny-ImageNet (55.16 / 20.54 with the 200-epoch teacher) — both `student_norm: False`. **CIFAR-10 has
no raw-$\ell_2$ cell**; its 84.66 / 51.87 comes from the *directional* variant, which also carries
`freeze_lr_epoch: 0.65`. The two designs were measured to be a tie, so this is presentational rather
than substantive. **Decision 2026-08-31: left as is** — the CIFAR-10 champion was produced on the
other server and is not being re-run. §3.2 should therefore say that the anchor is used unnormalized,
and note in the experimental setup that the CIFAR-10 cell uses the normalized variant, which the
design ablation shows to be equivalent (39.17 against 39.29 NRR on CIFAR-100, within the gap between
two schedule lengths of the same design).

### Claims scoped rather than dropped

"No hyperparameter" is stated in §3.2 as "no weight and no temperature" and attached to the
**backbone objective**, which is exactly what is true. The transfer claim is stated as "identical
configuration; only the dataset name and the teacher checkpoint change", which is verified: the
CIFAR-10 and CIFAR-100 champion configs are byte-identical up to the dataset string, and the
Tiny-ImageNet config differs only in the head flag and the checkpoint path.

### Deliberately absent from §3

The training radius $8.8/255$ against an evaluation radius of $8/255$ is reported in the
implementation paragraph rather than presented as a component, because it is held fixed across every
cell of the ablation ladder and is therefore not part of what is being claimed. `reformation` differs
between the CIFAR and Tiny-ImageNet configs but is only a fallback for `student_norm` / `teacher_norm`
when those are absent, and both are set explicitly everywhere, so it has no effect and is not
mentioned.
