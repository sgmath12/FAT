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
2020), IAD (Zhu et al., 2022), RSLAD (Zi et al., 2021), AdaAD (Huang et al., 2023), and IGDM (Lee et al.,
2025) differ in which teacher quantity is matched and where the teacher is evaluated, and together
they define the strongest reported accuracy–robustness frontiers for small architectures.

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
between the two must be tuned. DP-FAT (Cho & Kim, 2026) distills a naturally trained
teacher directly, in the fast single-step regime, and reports that the teacher's logit *magnitudes*
are the unreliable part: it rescales them per sample so that only the class-discriminative direction
is transferred. That is the closest existing use of a natural teacher, and the difference from ours
is what the teacher is asked for — DP-FAT repairs a logit target so that it can be trusted, where we
remove the logit target and place the anchor on the feature, which is where the magnitude question
does not arise. B-MTARD (Zhao et al., 2024) goes further still and uses a naturally trained network
explicitly to recover accuracy, pairing a clean teacher with a robust one so that each handles the
examples it is good at. That is the closest prior statement of the idea we pursue, and it
still requires the robust teacher — a WideResNet-70-16 on CIFAR-100, far larger than the student —
because the clean teacher supplies logits and cannot supply robustness. Both of its algorithmic
contributions, an entropy-based temperature balance and a dynamic loss balance, exist to reconcile
two teachers that push in opposite directions.

We do not claim that a naturally trained teacher is preferable to a robust one. A robust teacher is
better guidance, and obtaining it is precisely the expensive part; the question we take up is what
can be achieved when only the free teacher is available. The answer turns on **what the teacher is
asked to supply**. We ask it for standard accuracy alone and leave robustness entirely to the
student's inner maximization, and under that division the expected fragility does not materialize —
for a structural reason rather than an empirical one. Our objective matches the teacher's feature at
the *clean* input against the student's feature at the *adversarial* input, and it is the entire
objective for the backbone: the label term is removed rather than supplemented. The teacher is
therefore never evaluated inside the perturbation ball, so nothing it does there can enter the
optimization, and by the same token it cannot contribute robustness. All robustness is produced by
the student's inner maximization, exactly as in standard adversarial training. We further show that this single term is not merely a fidelity constraint. Writing $F$ for
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
since a robust teacher's own training is the expensive part. The non-monotonicity itself is not
peculiar to natural teachers: in the robust-teacher setting a more robust teacher is likewise
reported to fail to improve, or to harm, the student, with the teacher's predictive entropy on a
consistent subset of the data indicating which way it will go (Kim et al., 2026). Our ladder is the
natural-teacher analogue, and what varies along it is class separation rather than accuracy.

Building on these observations we propose **Clean Feature Anchoring (CFA)**, illustrated in Figure 1.
CFA trains a naturally supervised copy of the student architecture on the same data, initializes the
student from it, and then trains the backbone with the clean-feature anchor alone; the classifier is
inherited and never trained. To this we add **sensitivity-matched $\varepsilon$**, which redistributes
the per-sample attack radius in inverse proportion to the input sensitivity of the loss while holding
the total budget fixed, so that each example is perturbed to a comparable displacement of the
objective rather than a comparable displacement of the input. The resulting method has no
temperature, no loss weight and no auxiliary head, and we apply an identical configuration to every
dataset, changing only the dataset name and the teacher checkpoint.

On CIFAR-100 with ResNet-18, CFA reaches 62.65% standard accuracy at 28.77% AutoAttack accuracy.
Against the strongest method at comparable robustness this is **+5.29 points of standard accuracy**,
and against the strongest method at comparable standard accuracy it is **+4.81 points of AutoAttack
accuracy**; no method in this setting exceeds ours on both axes. The same configuration establishes
the best reported accuracy–robustness balance on CIFAR-10 and Tiny-ImageNet. We further run the
published distillation objectives on our own natural teacher, which is the teacher they would have in
this setting: ARD, RSLAD and AdaAD reach 20.24%, 21.30% and 23.19% AutoAttack accuracy, all below
plain adversarial training initialized at the same teacher (26.46%). The accuracy a natural teacher
holds is therefore not recovered by treating it as a logit target. Our contributions are as
follows:

- We show that a naturally trained network can guide adversarial training when it is asked only for
  standard accuracy. Evaluated at the clean input and used as the entire backbone objective, a
  feature anchor controls fidelity to the teacher and local stability of the student jointly, within
  a constant factor, as a single unweighted term — so the teacher's own fragility has no path into
  the optimization, and no coefficient trades the two properties off.

- We show that what a naturally trained teacher transfers is class geometry rather than accuracy,
  with teacher and student accuracy anti-correlated at $r = -0.999$ across teachers, which turns the
  teacher's training length into a trade-off control available only in this setting.

- We propose Clean Feature Anchoring together with sensitivity-matched $\varepsilon$, a method with
  no loss weight, no temperature and no per-dataset tuning, in which robustness remains entirely
  attributable to the student's own inner maximization.

- We demonstrate a new accuracy–robustness frontier on CIFAR-10, CIFAR-100 and Tiny-ImageNet, and
  show that the gain is not an artifact of weight averaging or adversarial weight perturbation: with
  weight averaging alone and half the training schedule, CFA already exceeds the best published
  result that uses both. Running the published distillation objectives on the same natural teacher
  places all of them below plain adversarial training from that teacher, which locates the
  difficulty in the logit target rather than in the teacher.

---

**Figure 1.** *(a) Conceptual diagram.* The teacher is evaluated only at the clean input $x$; the
student is evaluated at the adversarial input $x_{\mathrm{adv}}$, and the loss is the distance
between the two features. Because the teacher's $\varepsilon$-ball is never entered, its instability
is outside the objective, while the same distance simultaneously bounds the student's own variation
over the ball. *(b) Standard accuracy against AutoAttack accuracy* on CIFAR-100 with ResNet-18, for
adversarial training baselines, target-softening methods, and CFA.

---

## Writer's notes

**Numbers used.** The CIFAR-100 headline is **62.65 / 28.77 / NRR 39.43**
(`l2_bestrecipe_freezehead`, 2026-08-31): the unnormalized $\ell_2$ anchor with the classifier
frozen, which is exactly what §3 of the method describes. It supersedes 62.35 / 28.68 (same cell with
the head-KD term left on) and 62.17 / 28.59 (the *directional* variant). `abstraction.md` and
`writing.md` §1/§6 still carry the older pair and need the same update. The two designs were measured
to be a tie, so the change is one of consistency with the stated method, not of substance.

**Comparisons behind the two headline deltas.** Strongest at comparable robustness is ADR + WA + AWP
(57.36 / 28.50), giving clean +5.29 at AA +0.27. Strongest at comparable standard accuracy is
Generalist++ (62.97 / 23.96), giving AA +4.81 at clean −0.32 — stated as "comparable standard
accuracy" rather than "matched", since they remain 0.32 above us there. Both deltas grew with the head
freeze.

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
