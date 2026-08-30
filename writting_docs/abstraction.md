# Abstract

**Working title.** Anchor, Don't Imitate: Self-Distilled Feature Anchoring for the
Accuracy–Robustness Trade-off

**Placeholder method name.** SAFA — Self-Anchored Feature Adversarial training. (To be fixed;
used consistently below.)

---

## Abstract (main version, ~200 words)

Adversarial training buys robustness by giving up clean accuracy, and it does so with no teacher at
all: the robustness comes from the inner maximization, and the clean accuracy is simply lost.
Adversarial distillation recovers some of it, but by importing an already-robust network — typically
a large WideResNet adversarially trained with extra data — so the robustness originates outside the
student and the teacher costs more than the student does. We keep the teacherless setting and ask a
narrower question: can the *lost clean accuracy* be recovered from a copy of the student's own
architecture, trained naturally on the same data at no adversarial cost?

Such a teacher is the most accurate one available, but its accuracy rests on features an adversary
can flip, and copying it should import that fragility. We show it does not, provided the teacher is
used as a **feature anchor** rather than a logit target. Our objective matches the teacher's *clean*
feature at the *adversarial* input, so the teacher is never evaluated under attack and cannot
transfer its instability — nor, by the same token, any robustness. We prove the single term is
equivalent within a constant to fidelity plus local stability, and allocate the per-sample attack
radius by the loss's own input sensitivity at a fixed total budget.

Robustness therefore remains entirely the student's, while clean accuracy is recovered: on CIFAR-100
we improve clean accuracy by **4.8 points at matched AutoAttack**, and set a new
accuracy–robustness frontier on CIFAR-10, CIFAR-100 and Tiny-ImageNet. A naturally-trained teacher
further exposes a trade-off dial unavailable to robust-teacher distillation — teacher training length
— which moves the student along the frontier for free.

---

## Abstract (short version, ~150 words — for a page-limited venue)

Adversarial training gives up clean accuracy, and recovers none of it: robustness comes from the
inner maximization and no teacher is involved. Adversarial distillation recovers some, but by
importing an already-robust network, so the robustness originates outside the student. We keep the
teacherless setting and use a naturally-trained copy of the student's own architecture — free — to
recover only the lost clean accuracy. The key is to consume it as a **feature anchor** rather than a
logit target: matching the teacher's *clean* feature at the *adversarial* input means the teacher is
never evaluated under attack, so it can transfer neither its fragility nor any robustness. We prove
this objective is equivalent within a constant to fidelity plus local stability, and pair it with a
per-sample attack radius set by the loss's own input sensitivity at fixed total budget. Robustness
stays the student's own; clean accuracy improves by 4.8 points at matched AutoAttack on CIFAR-100,
with a new frontier on three datasets.

## Notes for the writer

**Robust-teacher distillation is a different problem, not a stronger result on ours.** ARD, RSLAD,
IAD, AdaAD, IGDM and B-MTARD assume an already-robust network exists — typically a WideResNet
adversarially trained with extra data — and ask how well that robustness transfers into a small
student. Their robustness *originates in the teacher* and their contribution is the transfer;
IGDM + AdaAD accordingly reaches 64.44 / 30.32 on ResNet-18 / CIFAR-100 from a BDM-AT WRN-28-10
teacher (72.58 / 38.83), and B-MTARD 65.08 clean from a WRN-70-16 robust teacher plus a clean one.

Ours is the teacherless setting: robustness is produced by the student's own inner maximization and
the teacher contributes none of it — §3 shows it is never evaluated under attack, so it *could* not.
The right phrasing is that the two assume different inputs and should be read side by side. Do
**not** write "we beat X" against that line, and do **not** write "a robust teacher is unnecessary",
which is a ranking claim we do not test. Within our own setting the strongest published AA is
ADR + WA + AWP at 57.36 / 28.50 and we are +4.81 clean at +0.09 AA.

**Comparison class, explicitly.** AT variants (PGD-AT, TRADES, MART), consistency and regularization
methods (Consistency-AT, RPAT), self-anchoring (ADR), and Generalist++.

**Terminology.** The per-sample radius rule was called "angular budget" in earlier drafts. It
equalizes the first-order movement of the **loss**, not an angle — for a cosine target that is
$\Delta\cos\theta$ and converting to $\Delta\theta$ carries a per-sample $1/\sin\theta$ — and it is
not specific to a directional objective. The correct name is **sensitivity-matched $\varepsilon$**.
The code knob keeps its old name `featdir_angeps_p` for log reproducibility.

**"Self-distillation" is used in the sense of reference [1]** (Cho et al., ICLR 2025): the teacher is
the same architecture trained on the same data, not an external or larger model. Ours differs in
that the teacher is trained *naturally* rather than adversarially, and is consumed as a feature
anchor rather than a logit target.

**Numbers cited.**

| | clean | AA | NRR |
|---|---:|---:|---:|
| CIFAR-100, ours | 62.17 | 28.59 | 39.17 |
| CIFAR-100, ADR + WA + AWP | 57.36 | 28.50 | 38.08 |
| CIFAR-10, ours | 84.66 | 51.87 | 64.33 |
| CIFAR-10, ADR + WA + AWP | 83.26 | 51.18 | 63.39 |
| Tiny-ImageNet, ours | 55.16 | 20.54 | 29.93 |
| Tiny-ImageNet, ADR + WA + AWP | 48.27 | 20.10 | 28.38 |

**Claims deliberately absent from the abstract.** That a natural teacher beats a robust one; that
AutoAttack improves (on CIFAR it does not — the gain is on the clean axis at matched robustness);
that the directional and raw feature targets differ (they tie). The teacher-length dial is stated as
an additional property, not as the headline, because its optimum is dataset-dependent and the
Tiny-ImageNet ladder has not peaked.
