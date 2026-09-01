# Abstract

**Working title.** Clean Feature Anchoring: Recovering Standard Accuracy in Adversarial Training
without a Robust Teacher

**Provisional method name.** CFA (Clean Feature Anchoring). Alternatives considered: SAFA
(Self-Anchored Feature Adversarial training), NFA (Natural Feature Anchoring).

---

## Abstract (main version)

Adversarial training is among the most reliable defenses against adversarial examples, but it degrades
standard accuracy, and this trade-off remains the central obstacle to its practical use. Adversarial
distillation mitigates the degradation with a teacher, but that teacher is adversarially trained and
typically costlier to obtain than the student, so the robustness reported originates outside the
student. We instead use a teacher of the same architecture trained naturally on the same data,
available at no adversarial cost, for the single purpose of restoring the accuracy that adversarial
training gives up. Such a teacher is fragile — under $\varepsilon=8/255$ its own feature rotates by
$63.8^\circ$ — and consuming it as adversarial distillation does makes matters worse: every published
objective given this teacher lands below simply not distilling. We show that the fragility is not
transferred when the teacher is used as a **feature anchor** instead. Matching the teacher's *clean*
feature at the *adversarial* input means the teacher is never evaluated under attack, so it transfers
neither its instability nor any robustness, and we prove that this single term is equivalent within a
constant factor to fidelity plus local stability, with no coefficient between them. Combined with a
per-sample attack radius set by the input sensitivity of the loss at fixed total budget, the method
has no loss weight, no temperature and no trained classifier, and transfers across datasets unchanged.
On CIFAR-100 with ResNet-18 it improves standard accuracy by **5.3 points at no cost in AutoAttack**,
and it sets a new accuracy–robustness frontier on CIFAR-10, CIFAR-100 and Tiny-ImageNet. Because the
teacher is naturally trained, its training length becomes a trade-off control that costs nothing to
exercise.

---

## Abstract (short version, for a page-limited venue)

Adversarial training improves robustness but degrades standard accuracy, and adversarial distillation
mitigates the trade-off with an adversarially trained teacher, so the robustness it reports originates
outside the student. We instead use a teacher of the same architecture trained naturally on the same
data, available at no adversarial cost, solely to restore the accuracy adversarial training gives up.
Such a teacher is fragile, and consuming it as adversarial distillation does makes matters worse:
every published objective given this teacher lands below simply not distilling. The fragility is not
transferred when the teacher is used as a **feature anchor** instead. Matching the teacher's *clean*
feature at the *adversarial* input means the teacher is never evaluated under attack, so it transfers
neither its instability nor any robustness; we prove this single term is equivalent within a constant
factor to fidelity plus local stability. Combined with a per-sample attack radius set by the input
sensitivity of the loss at fixed total budget, the method has no loss weight, no temperature and no
trained classifier. On CIFAR-100 with ResNet-18 it improves standard accuracy by 5.3 points at no cost
in AutoAttack, and sets a new frontier on three datasets.

---

## Notes for the writer

### Register

The reference abstracts state the trade-off in measured terms — *"degrades the standard accuracy"*
(ARREST), *"creates an inherent trade-off"* (RPAT), *"the performance towards clean examples is
negatively affected"* (B-MTARD). Earlier drafts of this abstract used *"buys robustness by giving up
clean accuracy"*, which is more forceful than the literature's own phrasing and reads as editorial.
None of the reference abstracts poses a rhetorical question; the drafts that did have been rewritten
as declaratives.

### ARREST is the closest prior work and must be positioned explicitly

ARREST (adversarial finetuning + representation-guided knowledge distillation + noisy replay) shares
both of our structural choices: it initializes from a standardly pretrained network, and its RGKD
term penalizes the distance between *the student's representation of an adversarial example* and
*the standardly pretrained network's representation of the clean example* — the same pairing as our
anchor, with angular distance as $d(\cdot)$.

What differs is what the term is *for*. In ARREST the constraint is added to the label cross-entropy,
$\mathcal L = \mathcal L_{\mathrm{CE}}(f(x+\delta),y) + \lambda\,\mathcal L_{\mathrm{RGKD}}$, so it
regularizes adversarial training and introduces $\lambda$, together with NR's threshold $\tau$ and
angle $\phi$. In ours the anchor **replaces** the cross-entropy: the backbone sees no label term at
all, the classifier is never trained, and there is no weight to set. That is also what makes
Proposition 1 available — the decomposition characterizes the objective only when the anchor *is* the
objective, not one term of a weighted sum.

Numerically, on CIFAR-10 / ResNet-18 neither dominates: ARREST 86.63 / 46.14 (NRR 60.21) against ours
85.58 / 51.79 (NRR 64.53). ARREST's CIFAR-100 results use WideResNet-34-10 (73.05 / 24.32) and are
not directly comparable to our ResNet-18 numbers.

### Robust-teacher distillation is a different problem, not a stronger result on ours

ARD, RSLAD, IAD, AdaAD, IGDM and B-MTARD assume an already-robust network exists — typically a
WideResNet adversarially trained with extra data — and study how well that robustness transfers into
a small student. Their robustness originates in the teacher and their contribution is the transfer.
IGDM + AdaAD accordingly reaches 64.44 / 30.32 on ResNet-18 / CIFAR-100 from a BDM-AT WRN-28-10
teacher (72.58 / 38.83), and B-MTARD reaches 65.08 clean using a WRN-70-16 robust teacher alongside a
clean one.

In our setting robustness is produced by the student's own inner maximization and the teacher
contributes none of it — it is never evaluated under attack, so it could not. The two settings assume
different inputs and are best read side by side. Do **not** write "we outperform" against that line,
and do **not** write "a robust teacher is unnecessary", which is a ranking claim we do not test.

### Related: teacher quality is not what transfers

The observation that a stronger teacher does not imply a better student has an independent parallel
in the adversarial-distillation literature (Toward Understanding Adversarial Distillation, ICML 2026:
*"a more robust teacher often fails to improve, or even harms, the student's robust generalization"*,
with the teacher's predictive entropy as an indicator of student robustness). Our teacher ladder is
the natural-teacher analogue: $r(\text{teacher clean},\ \text{student clean}) = -0.999$ across
teachers trained for 50–300 epochs, with class separation rather than accuracy as the transferred
quantity. Worth citing as convergent evidence rather than as a result we established first.

### Terminology

The per-sample radius rule was called "angular budget" in earlier drafts. It equalizes the
first-order movement of the **loss**, not an angle — for a cosine target that is $\Delta\cos\theta$,
and converting to $\Delta\theta$ carries a per-sample $1/\sin\theta$ — and it is not specific to a
directional objective. The correct name is **sensitivity-matched $\varepsilon$**. The code knob keeps
its old name `featdir_angeps_p` for log reproducibility.

"Self-distillation" is used in the sense of Cho et al. (ICLR 2025): the teacher is the same
architecture trained on the same data, not an external or larger model. Ours differs in that the
teacher is trained naturally rather than adversarially, and is consumed as a feature anchor rather
than a logit target.

### Numbers cited

| | clean | AA | NRR |
|---|---:|---:|---:|
| CIFAR-100, ours | 62.65 | 28.77 | 39.43 |
| CIFAR-100, ADR + WA + AWP | 57.36 | 28.50 | 38.08 |
| CIFAR-10, ours | 85.58 | 51.79 | 64.53 |
| CIFAR-10, ADR + WA + AWP | 83.26 | 51.18 | 63.39 |
| CIFAR-10, ARREST | 86.63 | 46.14 | 60.21 |
| Tiny-ImageNet, ours | 55.16 | 20.54 | 29.93 |
| Tiny-ImageNet, ADR + WA + AWP | 48.27 | 20.10 | 28.38 |

All ResNet-18, $\ell_\infty$, $\varepsilon = 8/255$; AA is AutoAttack on the full test set; NRR is the
harmonic mean of standard accuracy and AA.

### Claims deliberately absent

That a natural teacher outperforms a robust one; that AutoAttack accuracy improves (on CIFAR it does
not — the gain is on the standard-accuracy axis at matched robustness); that the directional and raw
feature targets differ (they tie). The teacher-length control is stated as an additional property
rather than a headline, since its optimum is dataset-dependent and the Tiny-ImageNet ladder has not
peaked.
