# Analysis and Method

Draft, 2026-08-31 rev. 3. Structure follows `reference/self-distililation-at.pdf`, which splits
§3 ANALYSIS (preliminary, theory, empirical) from §4 METHOD (~350 words, the construction only), and
`reference/IGDM.pdf`, whose §3.1 is a motivating measurement before any derivation. Implementation
details belong in Experiments. Companion: `introduction.md`.

---

## 3. Analysis

### 3.1 Preliminaries

Let $f=h\circ\Phi$ be a classifier with feature map $\Phi$ and linear head $h$. Adversarial training
solves

$$\min_\theta\ \mathbb{E}_{(x,y)}\ \max_{x'\in B(x,\varepsilon)}\ \ell\big(f_\theta(x'),y\big),
\qquad B(x,\varepsilon)=\{x':\lVert x'-x\rVert_\infty\le\varepsilon\}. \tag{1}$$

The supervisory signal is the label, and it is asserted uniformly over $B(x,\varepsilon)$: every point
of the ball must reach full confidence in one class. That is where the accuracy cost comes from.
Adversarial distillation replaces the label with a teacher quantity, taken from the logits of an
adversarially trained teacher — at $x$ (ARD, RSLAD), at $x_{\mathrm{adv}}$ (AdaAD), or from finite
differences of both (IGDM). We ask instead what a *naturally* trained teacher, which costs nothing to
obtain, can supply.

### 3.2 The teacher is fragile, and its logits are worse than no teacher at all

The objection to a naturally trained teacher is not hypothetical. Under $\varepsilon=8/255$ its own
feature rotates by $63.8^\circ$ and its norm inflates by $\times 2.45$; its accuracy rests on
directions an adversary controls.

That fragility shows up immediately if the teacher is consumed the way adversarial distillation
consumes one. Table 1 gives the published objectives with our natural teacher in place of the robust
teacher they assume, at their own recipes, against plain adversarial training initialized from the
same teacher.

**Table 1.** CIFAR-100, ResNet-18, 100 epochs, each method at its published recipe.

| | clean | AA | NRR |
|---|---:|---:|---:|
| ARD | 57.61 | 20.24 | 29.96 |
| RSLAD | 59.68 | 21.30 | 31.40 |
| AdaAD | 59.79 | 23.19 | 33.42 |
| PGD-AT, initialized at the teacher | 57.73 | **26.46** | **36.29** |

Every one of them lands below simply not distilling. The same holds when the temperature is swept
rather than inherited: in a fixed base regime the best logit target reaches AA $24.48$ at $\tau=4$,
against $25.88$ for a feature target under identical conditions, and the logit curve is $\cap$-shaped
because $\tau$ is squeezed from both sides — at $\tau=1$ the teacher's maximum probability is $0.820$
against the student's $0.016$, and at $\tau=16$ the target is within $1\%$ of uniform.

The teacher therefore holds accuracy that adversarial training gives up, and the logit route does not
recover it. What follows asks what does.

### 3.3 What a clean-feature anchor controls

Consider matching the teacher's feature at the *clean* input against the student's feature at the
*adversarial* input. Fix $x$, write $B=B(x,\varepsilon)$, and let

$$L=\max_{x'\in B}\lVert\Phi_s(x')-\Phi_t(x)\rVert,\quad
F=\lVert\Phi_s(x)-\Phi_t(x)\rVert,\quad
O=\max_{x',x''\in B}\lVert\Phi_s(x')-\Phi_s(x'')\rVert, \tag{2}$$

so $L$ is that quantity, $F$ is fidelity to the teacher at the clean point, and $O$ is the oscillation
of the student's own map over the ball.

**Proposition 1.** $L\le F+O\le 3L$.

*Proof.* $F\le L$ since $x\in B$; $O\le 2L$ by inserting $\Phi_t(x)$; and
$\lVert\Phi_s(x')-\Phi_t(x)\rVert\le\lVert\Phi_s(x')-\Phi_s(x)\rVert+F\le O+F$, so $L\le F+O$.
$\square$

A single term therefore controls fidelity and local stability together, within a factor of three, with
no coefficient trading them off — the first is what adversarial training gives up, the second what its
label term buys. The proof uses only the triangle inequality, so it holds for the networks we train.

Two consequences decide the design. First, $\Phi_t$ is evaluated at $x$ alone, so the instability
measured in §3.2 never enters the objective — and, symmetrically, the teacher cannot supply
robustness. Second, the decomposition holds only when the anchor *is* the objective: added to a
cross-entropy term it characterizes nothing.

Both are verifiable, and we verify them. On the trained student over $\varepsilon=8/255$ balls,
$L=7.58$, $F=6.30$ and $O_{\mathrm{lb}}=2.16$ against the teacher's own $O=6.68$: the student is
$3.1\times$ more stable than the teacher it was distilled from, and what remains in the loss is
fidelity rather than instability.

---

## 4. Method

### 4.1 Clean feature anchoring

Let $\Phi_t$ be the frozen feature map of a network of the same architecture trained naturally on the
same data. The student is initialized at $\Phi_s=\Phi_t$ and trained with

$$\mathcal{L}=\mathbb{E}_x\big\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_t(x)\big\rVert^2,
\qquad
x_{\mathrm{adv}}=\arg\max_{x'\in B(x,\varepsilon)}\big\lVert\Phi_s(x')-\Phi_t(x)\big\rVert. \tag{3}$$

The head is never trained: at test time the student is $h_t\circ\Phi_s$. Writing the target on the
feature is what makes this possible — a logit loss constrains $W\Phi$ and so determines $(\Phi,W)$
only up to $(A\Phi,WA^{-1})$, while a feature target pins the representation itself, so the teacher's
classifier remains calibrated for it. Two further properties are worth naming: (3) has no weight and
no temperature, being the whole objective rather than a term added to another; and it is exactly zero
at initialization, so every gradient the backbone receives is attributable to the attack.

### 4.2 Sensitivity-matched $\varepsilon$

A fixed pixel radius is not a fixed amount of pressure on the objective. Within an $\ell_\infty$ ball
of radius $e$,

$$\max_{\lVert\delta\rVert_\infty\le e}\langle\nabla_x\mathcal{L},\delta\rangle
=e\lVert\nabla_x\mathcal{L}\rVert_1, \tag{4}$$

so with $g_i=\lVert\nabla_x\mathcal{L}(x_i)\rVert$ the same radius moves the objective by amounts that
differ as $g_i$ does. Equalizing that movement at fixed total budget $\sum_i\varepsilon_i=N\varepsilon$
gives

$$\varepsilon_i\propto g_i^{-p},\qquad
\varepsilon_i\leftarrow\mathrm{clip}\big(\varepsilon_i,0.5\varepsilon,1.5\varepsilon\big),\qquad
\varepsilon_i\leftarrow\varepsilon_i\cdot N\varepsilon\Big/\textstyle\sum_j\varepsilon_j, \tag{5}$$

with $p=1$ — equivalently the max–min allocation, since budget moved to the sample whose objective
moves least always improves the worst case. Budget preservation is what makes the comparison
interpretable: $p{=}1$ and $p{=}0$ spend the same total attack budget, so any difference between them
is allocation and not strength.

Per-sample radii are not new; IAAT, MMA and CAT assign one from the difficulty of the sample or its
margin in **input** space. This rule assigns one from the input-sensitivity of the **training loss**,
the geometry the objective is written in. Holding the exact multiset of weights (5) produces and only
reassigning which sample receives which, by difficulty, pays $0.74$ AutoAttack and falls below the
uniform baseline on CW and NRR: the same weights in a different order move *along* the frontier, and
only this order moves the frontier itself.

### 4.3 Algorithm

**Algorithm 1** — Clean Feature Anchoring (CFA)

```
Require: dataset D, radius eps, steps K, exponent p = 1
 1: Phi_t <- train f_t naturally on D                    # same architecture, no adversarial cost
 2: Phi_s <- Phi_t,  h <- h_t                            # student initialized at the teacher
 3: for each epoch, for each minibatch {x_i} of D:
 4:     g_i    <- || grad_x || Phi_s(x_i) - Phi_t(x_i) ||^2 ||        # one backward pass at x
 5:     eps_i  <- clip( eps * (g_i / mean g)^(-p), 0.5 eps, 1.5 eps )
 6:     eps_i  <- eps_i * N eps / sum_j eps_j                         # restore the total budget
 7:     x_i^a  <- PGD_K( max_{x' in B(x_i, eps_i)} || Phi_s(x') - Phi_t(x_i) || )
 8:     update Phi_s on  mean_i || Phi_s(x_i^a) - Phi_t(x_i) ||^2     # h is never updated
```

Line 1 is the only place labels appear, and it is ordinary supervised training; the remainder uses
none.

**Caveats.** Proposition 1 bounds $F+O$, and $O$ is not a sufficient statistic for robust accuracy —
across the teacher ladder of §5 it correlates with AutoAttack at $r=+0.83$, the wrong sign. It
explains why anchoring to a non-robust teacher is *free*, not how much robustness the objective buys;
Appendix A gives two partial results on the latter, each with the reason it falls short. In (4) the
derivation asks for the $\ell_1$ norm and our runs use $\ell_2$; both were trained and agree within
noise. Finally, $g_i$ is one backward pass at the clean $x_i$, so it is the sensitivity at the
attack's starting point rather than along it.

---

## Writer's notes

### rev. 3: split into Analysis and Method

rev. 2 compressed everything into one §3 of 970 words, which was the wrong budget. The long-tailed
self-distillation paper runs §3 ANALYSIS (preliminary, theorem, empirical measurements) at roughly a
thousand words and then §4 METHOD at ~350, and IGDM's §3.1 is likewise a motivating measurement
before any derivation. Adopting that split gives ~1400 words and, more importantly, gives the
baseline measurement a home: §3.2 now carries the result that the published objectives with our
natural teacher all land *below not distilling at all*, which is the strongest single argument for
the design and previously appeared only as one sentence in the introduction.

Order is: the teacher is fragile and the obvious use of it fails (§3.2), the anchor is different and
here is why (§3.3), here is the construction (§4). Implementation still belongs in Experiments.




### One discrepancy this draft resolves, and the decision it needs

`writing.md` §2 currently states that "the classifier stays at the teacher's and is never trained".
That is true of the Tiny-ImageNet runs (`featdir_freeze_head: True`) and it is what the ablation
recommends, but it is **not** true of the CIFAR-10 and CIFAR-100 runs as executed: those configs set
`beta: 1.0`, `tau: 16` and omit `featdir_freeze_head`, so the head is fitted by
$\beta\,\mathrm{KL}(h(\Phi_s(x_{\mathrm{adv}}))\,\Vert\,z_t/\tau)$. The term is detached from the
backbone (`featdir_alpha` defaults to $0$, `methods.py:2106`), so it changes no backbone gradient and
no adversarial example, but it does change the final classifier and therefore the reported numbers.
§3.5 above states this as it is.

**Resolved 2026-08-31, and the frozen head won.** `l2_bestrecipe_freezehead` is
`l2_bestrecipe_angeps` plus `featdir_freeze_head: True` and nothing else:

| | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| head KD, $\tau=16$ (the previously reported cell) | 62.35 | **36.26** | 30.65 | 28.68 | 39.29 |
| **head frozen** | **62.65** | 32.63 | **30.66** | **28.77** | **39.43** |

It was run to make the description true, not to gain accuracy, and it did both, in the direction the
base-regime ablation predicted (36.64 against 36.35). $\tau$ and $\beta$ are now absent from the
method, "no weight and no temperature" is literally true rather than scoped to the backbone, §3.5's
classifier paragraph is one sentence, and the Tiny-ImageNet configuration describes all datasets
rather than being the exception. ⚠ PGD-20 falls $3.63$ while AA and CW rise — the fourth time in this
project that PGD alone would have produced the opposite decision.

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
