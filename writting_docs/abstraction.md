# Abstract

**Working title.** Anchor, Don't Imitate: Self-Distilled Feature Anchoring for the
Accuracy–Robustness Trade-off

**Placeholder method name.** SAFA — Self-Anchored Feature Adversarial training. (To be fixed;
used consistently below.)

---

## Abstract (main version, ~200 words)

Adversarial training buys robustness by giving up clean accuracy, and adversarial distillation is the
standard remedy. Existing methods transfer the teacher's *logits*, and their gains rest on a strong
*adversarially-trained* teacher — a cost that is often larger than training the student itself. We
ask whether the cheapest possible teacher suffices: a network of the **same architecture** trained
**naturally on the same data**, obtained by self-distillation at no adversarial cost. Such a teacher
is the most accurate one available, but its accuracy rests on features an adversary can flip, and
copying it should import that fragility.

We show it does not, provided the teacher is used as a **feature anchor** rather than a logit target.
Our objective matches the teacher's *clean* feature at the *adversarial* input, so the teacher is
never evaluated under attack and its own instability cannot enter the loss; we prove the single term
is equivalent within a constant to fidelity plus local stability. We further allocate the per-sample
attack radius by the loss's input sensitivity at a fixed total budget, which the geometry of the
objective — not sample difficulty — determines.

Among methods that do not require an adversarially-trained teacher, SAFA sets a new
accuracy–robustness frontier on CIFAR-10, CIFAR-100 and Tiny-ImageNet, improving clean accuracy by
**4.8 points at matched AutoAttack robustness** on CIFAR-100. A naturally-trained teacher also
exposes a trade-off dial that robust-teacher distillation does not have: teacher training length,
which moves the student along the frontier at no additional cost.

---

## Abstract (short version, ~150 words — for a page-limited venue)

Adversarial training trades clean accuracy for robustness, and adversarial distillation mitigates it
by transferring a teacher's logits — at the cost of first adversarially training a large teacher. We
show that a naturally-trained network of the same architecture, obtained for free by
self-distillation, is a sufficient teacher when used as a **feature anchor** instead of a logit
target. Matching the teacher's *clean* feature at the *adversarial* input means the teacher is never
evaluated under attack, so its fragility cannot be inherited; we prove this objective is equivalent
within a constant factor to fidelity plus local stability. Combined with a per-sample attack radius
allocated by the loss's own input sensitivity at fixed total budget, our method sets a new
accuracy–robustness frontier among approaches that need no adversarially-trained teacher, improving
clean accuracy by 4.8 points at matched AutoAttack on CIFAR-100, with consistent gains on CIFAR-10
and Tiny-ImageNet.

---

## Notes for the writer

**Positioning is load-bearing and must not be overstated.** IGDM + AdaAD reports 64.44 clean /
30.32 AA on ResNet-18 / CIFAR-100, which dominates our 62.17 / 28.59 on both axes. It uses a BDM-AT
WideResNet-28-10 teacher (72.58 / 38.83), i.e. an adversarially-trained teacher larger than the
student. Every claim of "new frontier" must therefore carry **"among methods that do not require an
adversarially-trained teacher."** Within that class the strongest published AA is ADR + WA + AWP at
57.36 / 28.50, and we are +4.81 clean at +0.09 AA.

**Comparison class, explicitly.** AT variants (PGD-AT, TRADES, MART), consistency and
regularization methods (Consistency-AT, RPAT), self-anchoring (ADR), and multi-teacher methods whose
robust half we do not use. **Not** ARD / RSLAD / IAD / AdaAD / IGDM, which all require an
adversarially-trained teacher, nor B-MTARD, which needs both a clean and a robust one.

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
