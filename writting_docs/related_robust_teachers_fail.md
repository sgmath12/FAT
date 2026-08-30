# Why Robust Teachers Fail (ICML 2026) — mechanism, and what it implies for us

Read 2026-08-30 from `writting_docs/reference/21950_Toward_Understanding_Adv.pdf`
("Toward Understanding Adversarial Distillation: Why Robust Teachers Fail", KAIST, PMLR 306, 2026).
⚠ The filename looks like an OpenReview submission id; confirm the published citation before use.

---

## 1. Their mechanism

**Robustly unlearnable set $S_U$.** A consistent subset of training samples is misclassified by
*every* adversarially-trained model, identified by intersecting 60 models (6 training paradigms × 10
seeds) at their peak-robust-accuracy epochs. It is **model-dependent, not data-dependent**
(CIFAR-10, intersection column):

| architecture | unlearnable | learnable |
|---|---:|---:|
| MobileNet-V2 | 8,979 | 19,385 |
| ResNet-18 | 5,217 | 21,899 |
| WRN-28-10 | 1,697 | 19,610 |
| WRN-34-10 | 1,559 | 16,397 |

Larger models learn a wider portion robustly. Their reading: robust classification depends on robust
features whose *representability* varies with capacity, so some samples rely on robust features a
small student cannot represent.

**The failure path.** Once the learnable feature $u$ is acquired, learnable-sample gradients decay
(Lemma 4.9) and the only residual source of updates is $S_U$. The student cannot fit those through
the unrepresentable feature $v$, so it fits them by **memorizing sample-specific noise**
(Lemma 4.11). That memorized direction then serves as an admissible adversarial perturbation on
fresh test points, and robust test error goes to $\tfrac12-o(1)$ (Theorem 4.7). **This already
happens in plain AT** — it is their account of robust overfitting, not only of distillation failure.

**Where the teacher enters.** The AD objective is a soft-target cross-entropy weighted by the
teacher's own confidence:

$$L_{AD}=\sigma\!\big(y f_T(X)\big)\,\ell\!\big(y f_W(\tilde X)\big)+\sigma\!\big(-y f_T(X)\big)\,\ell\!\big(-y f_W(\tilde X)\big).$$

- **Bad Teacher** — confident on $S_U$ ($y f_T\ge\Gamma$, with $\Gamma\ge\tilde\Omega(d)$ so the
  sigmoid is saturated): the weights collapse to one-hot, the objective becomes effectively hard-label,
  the residual gradient persists, and noise memorization proceeds (Theorem 4.8).
- **Good Teacher** — uncertain on $S_U$ ($y f_T=0$): both weights are $\tfrac12$, the two terms
  oppose, the residual gradient is suppressed, and noise responses stay at initialization scale.

**Both teachers are robust.** In their words, the Bad Teacher "is not *bad* in isolation; rather, it
has access to the additional robust feature $v$, which the capacity-constrained student cannot
represent." A *stronger* teacher is worse precisely because it is confident about something the
student is structurally blind to.

**Practical criterion.** The teacher's predictive entropy on $S_U$ predicts student robustness, and
is offered as an a-priori teacher-selection rule.

---

## 2. What it implies for us

### 2.1 It explains our temperature sweep

Their criterion says an effective teacher must be **uncertain** where the student cannot follow. Our
natural teacher is the opposite extreme — measured on clean CIFAR-100 test data:

| | mean max-prob | predictive entropy | $\lVert z\rVert$ |
|---|---:|---:|---:|
| natural teacher | **0.820** | **0.76** | 16.63 |
| trained robust student | 0.016 | 4.60 | 0.81 |

(uniform over 100 classes has entropy $\ln 100 = 4.605$.)

By their criterion a maximally-confident teacher is the worst possible logit teacher, and that is
what we measure: no temperature reaches the feature anchor (base regime, CIFAR-100, raw, 50ep),

| | clean | AA | NRR |
|---|---:|---:|---:|
| logits, $\tau=1$ | 58.26 | 20.84 | 30.70 |
| logits, $\tau=4$ | 59.39 | **24.48** | **34.67** |
| logits, $\tau=16$ | 57.78 | 24.00 | 33.91 |
| **feature anchor, no head KD** | **62.72** | **25.88** | **36.64** |

**$\tau$ is exactly the knob that converts a Bad Teacher into a Good one** — raising it injects the
uncertainty their theorem asks for, and the teacher's max-prob falls $0.820\to0.154\to0.0196$. Our
∩-shaped curve is the statement that the conversion cannot be completed: the same $\tau$ that buys
uncertainty destroys the target ($0.0196$ against uniform $0.0100$) and displaces the warm start by a
factor of $\tau$. **Their framework says why uncertainty is what is needed; our sweep says why
temperature cannot supply it.** These are complementary and should be cited that way.

### 2.2 Our objective has no channel for teacher confidence

Their mechanism runs entirely through $\sigma(y f_T)$, a sigmoid that **saturates** when the teacher
is confident and thereby converts soft targets into hard-label gradients. Our objective is

$$\big\lVert\Phi_s(x_{\mathrm{adv}})-\Phi_t(x)\big\rVert^2,$$

an $L_2$ regression whose gradient is **linear in the residual and never saturates**. There is no
$\sigma(\cdot)$ in it, so teacher confidence has no route by which to become hard-label pressure.
This is a structural difference, not a matter of tuning.

There is a second, sharper reading. A **classification** loss instructs the model to make the margin
large *by any available means*; when the intended feature is unrepresentable, noise is an available
means, which is Lemma 4.11. A **regression** loss instructs the model to match a specific vector, and
the component it cannot represent simply remains unmatched — there is no incentive to reach the
target by another route. ⚠ Stated as an argument, not a result: "unrepresentable component
contributes no gradient" is exact for a linear map and loose for a network.

### 2.3 Same-architecture teaching removes their premise, but only halfway

Their setup requires a capacity gap: the teacher represents a robust feature $v$ the student cannot.
Our teacher is the **same ResNet-18** as the student, so there is no capacity gap in that direction.

⚠ This must not be overclaimed. Their $S_U$ is defined with respect to *adversarially-trained*
models, and a *naturally*-trained ResNet-18 does represent things a robust ResNet-18 will not use —
the non-robust features. So a mismatch exists; it is simply a different one, and it is the one
Proposition 1 addresses: $\Phi_t$ is evaluated at $x$ and nowhere else, so the non-robust route never
enters the objective.

### 2.4 Convergent, not prior

Our teacher ladder (METHOD.md §9) reports $r(\text{teacher clean},\ \text{student clean})=-0.999$
across teachers trained 50–300 epochs — a more accurate teacher yielding a worse student. That is the
natural-teacher analogue of their "a more robust teacher often fails to improve, or even harms, the
student". **Cite as convergent evidence; do not claim priority.** The mechanisms differ: theirs is
confidence-driven noise memorization on an unlearnable set; ours is the student's inability to
reproduce a widening class separation ($\cos(\hat\Phi_s,\hat\Phi_t)$ falls $0.8607\to0.8311$ and
clean accuracy tracks it at $r=+0.994$).

---

## 3. Testable prediction, and what we found

### 3.1 The direct transport fails, and the reason is informative

**Prediction.** If teacher confidence on samples the student cannot handle is what does the damage,
then the teacher should be *more* confident on the samples our student loses under attack.

**Measured** (CIFAR-100 test, matched CE-PGD on each student, teacher read on clean $x$; 20 batches):

| student | robust acc | $H_{\text{fail}}$ | $H_{\text{ok}}$ | $\Delta H$ | $p_{\text{fail}}$ | $p_{\text{ok}}$ |
|---|---:|---:|---:|---:|---:|---:|
| feature anchor | 29.30% | 1.011 | 0.156 | +0.855 | 0.759 | 0.967 |
| logit distill $\tau=16$ | 32.30% | 1.028 | 0.202 | +0.825 | 0.756 | 0.954 |
| logit distill $\tau=4$ | 30.39% | 1.015 | 0.180 | +0.835 | 0.759 | 0.961 |
| logit distill $\tau=1$ | 23.32% | 0.945 | 0.155 | +0.790 | 0.776 | 0.966 |

**The sign is opposite and the measure does not discriminate.** The teacher is markedly *less*
confident where the student fails (entropy $1.01$ against $0.16$), and $\Delta H$ is the same
$0.79$–$0.86$ for all four students. It is re-measuring sample difficulty, not anything about how the
teacher was consumed.

This does not contradict their result. Their Bad Teacher is confident on samples that are
*robustly* unlearnable, which is a property defined against adversarially-trained models; a
naturally-trained teacher's confidence tracks *clean* difficulty instead, and clean difficulty and
robust unlearnability are not the same partition. **A natural teacher is, by this measure,
automatically uncertain where the student struggles** — which is the Good Teacher condition, obtained
for free rather than by selection. ⚠ It is uncertain only relatively: $p_{\text{fail}}=0.759$ is
still a confident prediction in absolute terms, so this should not be written as "our teacher
satisfies their criterion".

### 3.2 The observable consequence — robust overfitting — does discriminate

Their mechanism predicts *robust overfitting*: the residual classification gradient on samples the
student cannot fit is discharged into memorized noise, so robust test accuracy peaks and then
declines. The anchored objective has **no classification loss on the backbone at all**, hence no
classification residual to discharge, which predicts a later peak and a smaller decline.

The controlled comparison is `at_teacherinit_matched` against the anchored cells: identical
regime — 100 epochs, AdamW $0.021$, OneCycle, 10-step PGD, $\varepsilon_{\mathrm{tr}}=8.8/255$, WA,
AWP proxy, initialized at the same teacher — differing only in whether the backbone is trained by
label cross-entropy or by the feature anchor.

| cell | peak PGD-20 | final | drop | peak epoch |
|---|---:|---:|---:|---:|
| **teacher-init label CE** | 32.78 | 31.66 | **$-1.12$** | **55** |
| anchor, champion regime | 35.14 | 34.94 | $-0.20$ | 90 |
| anchor + sensitivity-matched $\varepsilon$ | 34.97 | 34.77 | $-0.20$ | 95 |
| anchor, L2 target, best recipe | 36.64 | 36.26 | $-0.38$ | 85 |
| pure KD, champion regime | 35.34 | 35.10 | $-0.24$ | 90 |

**The label-CE control peaks 35 epochs earlier and declines $5.6\times$ further**, under machinery
(WA and AWP) specifically known to suppress robust overfitting. Among the anchored cells the decline
is $0.20$–$0.38$ and the peak sits at 85–95 of 100 epochs.

A second, weaker instance appears in the 50-epoch base sweep, where the schedule is too short for
robust overfitting to develop and every cell peaks at the last measured point (epoch 45) — except
$\tau=1$, which peaks at 35 and declines $1.71$. That is the cell whose target is closest to a hard
label, which is the configuration their theorem indicts.

⚠ **What this is and is not.** It is consistent with their mechanism and with the structural reading
in §2.2, and the regime is controlled. It is not a test of *their* claim: we have not identified an
unlearnable set, we do not vary teacher confidence at fixed everything else, and "no classification
residual" is an argument about the objective rather than a measurement of noise memorization. The
honest statement is that the objective which removes the classification term also removes most of
the robust overfitting, in a comparison where nothing else changes.

---

## 4. The controlled comparison, completed

`at_teacherinit_matched` finished 2026-08-30. It is the anchored cell with one substitution — the
backbone trained by $\mathcal L_{\mathrm{CE}}(f(x_{\mathrm{adv}}),y)$ instead of the feature anchor —
at an otherwise identical regime (100ep, AdamW $0.021$, OneCycle, 10-step PGD,
$\varepsilon_{\mathrm{tr}}=8.8/255$, WA, AWP proxy, same teacher checkpoint for initialization).

| | clean | PGD-20 | CW | AA | NRR | $\cos$ vs teacher |
|---|---:|---:|---:|---:|---:|---:|
| label CE, teacher-init | 57.73 | 31.30 | 28.43 | 26.46 | 36.28 | **0.4295** |
| feature anchor (L2, best recipe) | **62.35** | **36.26** | **30.65** | **28.68** | **39.29** | **0.8344** |
| difference | +4.62 | +4.96 | +2.22 | +2.22 | +3.01 | +0.4049 |

Three things follow.

**The initialization does not carry the result.** Starting at the clean teacher and training with
label cross-entropy lands at 57.73, inside the published adversarial-training band (PGD-AT 56.56,
RPAT 58.22, Consistency-AT 58.53), and retains only $\cos=0.43$ of the teacher. The anchor is what
holds the representation and what buys the accuracy.

**The gain is not confined to the clean axis here.** Against this control the anchor is ahead on
*both* axes (+4.62 clean, +2.22 AA). The paper's usual signature — AA tie, clean gain — is measured
against *published methods at their own operating points*; against label-CE adversarial training in
our own regime the anchor wins outright. Both statements should be reported, with the difference in
comparison basis made explicit.

**Robust overfitting separates by objective, not by regularization.** Both cells carry WA and AWP,
machinery whose purpose is to suppress robust overfitting, and the label-CE cell still peaks 35
epochs earlier and declines $5.6\times$ further. That is the observable consequence §3.2 predicted
from the absence of a classification term on the backbone.
