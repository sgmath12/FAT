# Introduction

Draft, 2026-08-30. Register matched to `reference/IGDM.pdf`, `reference/self-distililation-at.pdf`
and `reference/ADR.pdf`: problem, what has been done, the limitation that remains, what we found,
what we propose, contributions. No rhetorical questions; the trade-off is stated in measured terms.

---

## 1. Introduction

Deep neural networks are vulnerable to adversarial examples, imperceptible perturbations of the
input that change the prediction (Szegedy et al., 2014; Goodfellow et al., 2015; Carlini & Wagner,
2017), which raises concerns wherever such models are deployed in safety-critical systems
(Grigorescu et al., 2020; Ma et al., 2021). Among the defenses proposed in response, adversarial
training (Madry et al., 2018) remains the most reliable (Athalye et al., 2018; Croce & Hein, 2020):
the model is optimized on perturbations produced by an inner maximization, and the robustness it
attains has survived a decade of stronger attacks. Its cost, however, is standard accuracy. A
network trained adversarially classifies unperturbed inputs considerably worse than the same network
trained naturally, and the gap grows with the perturbation budget. This accuracy–robustness
trade-off is partly intrinsic to the objective (Tsipras et al., 2019; Zhang et al., 2019) and it is
the principal obstacle to deploying adversarially trained models in practice.

Two lines of work reduce it. The first modifies the target inside adversarial training. Hard labels
are a poor supervisory signal for perturbed inputs, since they assert that every point of an
$\varepsilon$-ball belongs to one class with full confidence, and softening them alleviates both the
trade-off and robust overfitting (Rice et al., 2020). Label smoothing (Pang et al., 2021),
consistency regularization (Dong et al., 2022), and annealing self-distillation rectification (Wu et
al., 2024) all replace the one-hot target with a smoother one, the last using the model's own weight
average as the source. These methods require no additional network. The second line is adversarial
distillation, which supplies the target from an adversarially trained teacher: ARD (Goldblum et al.,
2020), IAD (Zhu et al., 2022), RSLAD (Zi et al., 2021), AdaAD (Huang et al., 2023), IGDM (Lee et al.,
2025) and B-MTARD (Zhao et al., 2023) differ in which teacher quantity is matched and where the
teacher is evaluated, and together they define the strongest reported accuracy–robustness frontiers
for small architectures.

The two lines leave a gap between them. Adversarial distillation presupposes that an already-robust
network exists, and in practice that network is a WideResNet adversarially trained with large
volumes of generated data (Wang et al., 2023) — costlier to obtain than the student it teaches. The
robustness such methods report therefore originates outside the student, and the contribution being
measured is the quality of the transfer. The teacher that is genuinely free, a naturally trained
copy of the student's own architecture, is not used as a source of guidance for the adversarial term,
and the reason is well founded: a naturally trained network attains its accuracy through features an
adversary can manipulate (Ilyas et al., 2019), so transferring what it computes is expected to
transfer that fragility as well. ARREST (Suzuki et al., 2023) comes closest to using one — it
initializes the student from a standardly pretrained network and penalizes the angular distance
between the student's representation of an adversarial example and the pretrained network's
representation of the clean example — but there the penalty is added to the label cross-entropy as
one weighted term among several, so the backbone is still driven by hard labels and the balance
between the two must be tuned.

We find that the expected fragility does not materialize, and that what determines whether it does is
not the teacher's robustness but **the input at which the teacher is evaluated**. Our objective
matches the teacher's feature at the *clean* input against the student's feature at the *adversarial*
input, and it is the entire objective for the backbone: the label term is removed rather than
supplemented. Under that construction the teacher is never queried off the data manifold, so its own
instability cannot enter the optimization, and it cannot contribute robustness either — all
robustness is produced by the student's inner maximization, exactly as in standard adversarial
training. We further show that this single term is not merely a fidelity constraint. Writing $F$ for
the student's clean deviation from the teacher and $O$ for the diameter of the student's own output
over the $\varepsilon$-ball, the anchor loss $L$ satisfies $L \le F + O \le 3L$, so minimizing it
controls teacher fidelity and local stability together, within a constant factor, with no weight to
balance them. The bound uses only the triangle inequality and therefore holds for the networks we
actually train.

A second finding concerns what the teacher supplies. Across a ladder of naturally trained teachers
differing only in training length, teacher accuracy and student accuracy are almost perfectly
*anti*-correlated ($r = -0.999$): the most accurate teacher produces the worst student. What the
student inherits is the teacher's class geometry rather than its predictions, and the two are not
monotone in one another. This makes the teacher's training length a control on the
accuracy–robustness frontier that costs nothing to exercise and that a robust teacher does not offer,
since a robust teacher's own training is the expensive part.

Building on these observations we propose **Clean Feature Anchoring (CFA)**, illustrated in Figure 1.
CFA trains a naturally supervised copy of the student architecture on the same data, initializes the
student from it, and then trains the backbone with the clean-feature anchor alone; the classifier is
inherited and never trained. To this we add **sensitivity-matched $\varepsilon$**, which redistributes
the per-sample attack radius in inverse proportion to the input sensitivity of the loss while holding
the total budget fixed, so that each example is perturbed to a comparable displacement of the
objective rather than a comparable displacement of the input. The resulting method has no
temperature, no loss weight and no auxiliary head, and we apply an identical configuration to every
dataset, changing only the dataset name and the teacher checkpoint.

On CIFAR-100 with ResNet-18, CFA reaches 62.35% standard accuracy at 28.68% AutoAttack accuracy.
Against the strongest method at comparable robustness this is **+4.99 points of standard accuracy**,
and against the strongest method at comparable standard accuracy it is **+4.72 points of AutoAttack
accuracy**; no method in this setting exceeds ours on both axes. The same configuration establishes
the best reported accuracy–robustness balance on CIFAR-10 and Tiny-ImageNet. Our contributions are as
follows:

- We identify that the obstacle to guiding adversarial training with a naturally trained network is
  not the teacher's fragility but the input at which the teacher is evaluated, and we show that a
  clean-feature anchor evaluated at the adversarial input controls teacher fidelity and local
  stability jointly, within a constant factor, as a single unweighted term.

- We show that what a naturally trained teacher transfers is class geometry rather than accuracy,
  with teacher and student accuracy anti-correlated at $r = -0.999$ across teachers, which turns the
  teacher's training length into a trade-off control available only in this setting.

- We propose Clean Feature Anchoring together with sensitivity-matched $\varepsilon$, a method with
  no loss weight, no temperature and no per-dataset tuning, in which robustness remains entirely
  attributable to the student's own inner maximization.

- We demonstrate a new accuracy–robustness frontier on CIFAR-10, CIFAR-100 and Tiny-ImageNet, and
  show that the gain is not an artifact of weight averaging or adversarial weight perturbation:
  with weight averaging alone and half the training schedule, CFA already exceeds the best published
  result that uses both.

---

**Figure 1.** *(a) Conceptual diagram.* The teacher is evaluated only at the clean input $x$; the
student is evaluated at the adversarial input $x_{\mathrm{adv}}$, and the loss is the distance
between the two features. Because the teacher's $\varepsilon$-ball is never entered, its instability
is outside the objective, while the same distance simultaneously bounds the student's own variation
over the ball. *(b) Standard accuracy against AutoAttack accuracy* on CIFAR-100 with ResNet-18, for
adversarial training baselines, target-softening methods, and CFA.

---

## Writer's notes

**Numbers used, and one inconsistency to resolve.** The CIFAR-100 headline here is
**62.35 / 28.68 / NRR 39.29** (`l2_bestrecipe_angeps`, reproduced bit-identically on 2026-08-30 by
`ladder_angeps_waawp_100ep`). `abstraction.md` and `writing.md` §1/§6 still quote **62.17 / 28.59 /
39.17**, which is the *directional* variant. The two designs were measured to be a tie, but §2 of
`writing.md` states the objective as an unnormalized $\ell_2$ anchor, so the raw number is the one
consistent with the stated method and the other two documents should be updated to match.

**Comparisons behind the two headline deltas.** Strongest at comparable robustness is ADR + WA + AWP
(57.36 / 28.50), giving clean +4.99 at AA +0.18. Strongest at comparable standard accuracy is
Generalist++ (62.97 / 23.96), giving AA +4.72 at clean −0.62 — stated as "comparable standard
accuracy" rather than "matched", since they are 0.62 above us there.

**The fourth contribution bullet** cites the 2026-08-30 ladder: 50 epochs with weight averaging only
reaches NRR 38.46, above ADR + WA + AWP at 38.08 over 100 epochs. It is included because "the gains
come from the stack" is the most likely first objection, and this answers it in one sentence.

**Robust-teacher distillation is deliberately not ranked against.** Paragraph 3 states that their
robustness originates in the teacher, and paragraph 4 states that ours originates in the student;
the two are presented as different assumptions rather than as a comparison. IGDM + AdaAD reaches
64.44 / 30.32 on this architecture from a BDM-AT WRN-28-10 teacher, above us on both axes, and the
text must never imply otherwise. The measurement that belongs in the experiments section instead is
the four published distillation objectives run on *our* natural teacher, which is what those methods
would have available in this setting.

**ARREST placement.** Positioned at the end of paragraph 3 as the closest prior work, with the
distinction stated as structural — the anchor is one weighted term added to cross-entropy there, and
the whole backbone objective here — rather than as a numerical claim, since on CIFAR-10 neither
method dominates (ARREST 86.63 / 46.14, ours 84.66 / 51.87).

**Citations to fill.** Grigorescu et al. 2020 and Ma et al. 2021 are the safety-critical citations
used by both KAIST reference papers; keep or drop them together. Wang et al. 2023 is "Better
Diffusion Models Further Improve Adversarial Training", the source of the BDM-AT teacher those
methods use.
