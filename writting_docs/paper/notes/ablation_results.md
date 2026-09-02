# Ablation results, as they land

CIFAR-100 / ResNet-18. Base regime unless stated: teacher warm start, no weight averaging, no AWP,
$p = 0$, `train_eps` 8.8/255. Numbers are last-epoch, AutoAttack (`apgd-ce` + `apgd-t`).

---

## C1 — read the teacher at $x_{\mathrm{adv}}$ instead of $x$ *(done, 2026-09-02)*

`abl_teacher_at_adv`, 50 epochs, against `ladder_p0_50ep` = the same config with the flag off.

| | clean | FGSM | PGD-20 | CW | **AA** |
|---|---:|---:|---:|---:|---:|
| teacher read at $x$ (`ladder_p0_50ep`) | 61.33 | — | — | — | **26.19** |
| teacher read at $x_{\mathrm{adv}}$ | **76.11** | 10.32 | 0.00 | 0.00 | **0.00** |
| *(the teacher itself)* | 77.66 | — | — | — | 0.00 |

**Robustness does not degrade — it is gone.** Not 20, not 5. Exactly zero under every attack, with
clean accuracy 1.55 below the natural teacher's. Moving the read point does not weaken the objective;
it destroys it, because $\lVert\Phi_s(x') - \Phi_t(x')\rVert$ is minimized by $\Phi_s = \Phi_t$, and
that solution is a natural model. The training loss confirms the objective became trivial: 6.26 at
epoch 49 against roughly 50.7 for the anchor, an eight-fold collapse in the value being minimized.

This is the strongest available evidence for §3.3. Evaluating the teacher at the clean point is not
an implementation detail; it is the entire reason the objective has a non-trivial solution. It also
settles the framing question directly — **the anchor is load-bearing for robustness**, not merely
compatible with it.

⚠ **The recorded prediction was half wrong, and the error is informative.** It said "clean falls and
AA falls with it, because the target itself is now unstable." AA fell to the floor, but clean *rose*
by 14.78, to just under the teacher. The prediction treated a moving target as a source of noise the
student would fail to track. What actually happens is the opposite: the target becomes trivially
trackable, the student tracks it perfectly, and inherits its zero robustness. The mechanism is
degeneracy, not instability, and §3.3 should say degeneracy.

---

## C2 — does the anchor buy robustness by itself *(done, 2026-09-02)*

`abl_ce_nostack`: label cross-entropy on $x_{\mathrm{adv}}$ at the anchor's own base regime — same
teacher initialization, same 100 epochs, same `train_eps`, no WA, no AWP. The comparison the paper
was previously making across two different stacks, now made across one.

| | clean | PGD-20 | CW | **AA** | NRR | Avg |
|---|---:|---:|---:|---:|---:|---:|
| anchor (`ladder_p0_100ep`) | **61.21** | **31.26** | **26.82** | **25.24** | **35.74** | **43.23** |
| label CE, same regime (`abl_ce_nostack`) | 54.60 | 20.60 | 21.21 | 19.45 | 28.68 | 37.02 |
| **anchor − CE** | **+6.61** | **+10.66** | **+5.61** | **+5.79** | **+7.06** | **+6.21** |
| *(label CE with WA + AWP, `at_teacherinit_matched`)* | 57.73 | — | — | 26.46 | 36.29 | 42.09 |

The pre-registered read was: AA $\geq 25.24$ means the anchor buys no robustness on its own. It came
in at **19.45**, so the other branch holds — **the anchor buys 5.79 AutoAttack points by itself**,
with no sensitivity-matched $\varepsilon$, no weight averaging and no AWP, while simultaneously
buying 6.61 clean. Both axes, from the loss alone.

**And it is not an artifact of cross-entropy overfitting.** The obvious objection is that CE without
weight averaging robustly overfits at 100 epochs, so the comparison is unfair. It does overfit — PGD
peaks at 23.31 around epoch 70 and decays to 21.13. The anchor does not: its PGD rises monotonically
to the last evaluation, 25.23 → 27.23 → 28.86 → 29.12 → 30.72 → 31.37 → **31.47**. Even against CE's
*best* epoch the anchor's *final* is +8.16 PGD. Not robustly overfitting where cross-entropy does is
itself a property of the objective, and it is the property weight averaging is normally bought to
supply.

**What this does to the paper's framing.** The safe claim was "the anchor keeps accuracy and does not
cost robustness; WA and AWP supply the robustness." That understates the measurement. At matched
stack the anchor supplies most of it: +5.79 AA over cross-entropy, against the +2.40 that separates
the shipped recipe from cross-entropy *with* WA and AWP. The stack's marginal contribution is real
but smaller than the anchor's. §2.3 should be rewritten around C1 and C2 together — the anchor is a
source of robustness, the clean read point is why, and the stack is additive on top.
