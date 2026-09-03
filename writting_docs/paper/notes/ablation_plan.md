# Ablation structure — what each claim rests on

Written 2026-09-01 from an inventory of every run in `results/`: **158 logs, 85 with AutoAttack**
(CIFAR-100 70, CIFAR-10 15; July 10, August 67, September 8). The point of this document is to map
each claim the paper makes onto the runs that support it, so the ablation section is built from what
we measured rather than from what is convenient to show.

Cells marked ⚠ are **missing** and are the ones worth spending GPU time on.

---

## The claims, and what each needs

### C1 — The teacher may be non-robust because it is read at $x$ and nowhere else

This is the central structural claim (§3.3, Prop. 1). Everything else in the paper is downstream of
it, and **it has never been tested directly.**

| evidence | status |
|---|---|
| Prop. 1, $L \le F+O \le 3L$ | proved |
| $L/F/O$ measured: student $3.1\times$ more stable than its teacher | have |
| Teacher's own instability: $63.8^\circ$ rotation, $\times2.45$ norm | have |
| Published objectives that read the teacher off $x$ collapse (AdaAD, IGDM) | have (§5.3) |
| ⚠ **Our own objective with the teacher read at $x_{\mathrm{adv}}$** | **MISSING** |
| ⚠ **Label CE at the anchor's own base regime** | **MISSING** |
| ⚠ **The anchor with a label-CE attack** | **MISSING** |

The last row is the experiment. Everything currently supporting C1 is either a proof about our
objective or a measurement of *someone else's* method, and a referee will say the two are confounded:
AdaAD differs from us in the read point **and** in reading logits rather than features. The knob
`featdir_teacher_at_adv` was added today (`utils.inner_featdir_teacher_at_adv`); it moves the target
in both the attack and the loss and changes nothing else.

**Prediction, stated before the run.** Clean accuracy falls and AA falls with it, because the target
itself is now unstable; and $O$ measured on the resulting student rises toward the teacher's $6.68$
rather than sitting at $2.16$. If instead it ties, C1 is wrong as an *explanation* even though the
method still works, and §3.3 must be rewritten as a description rather than a mechanism.

**Cost.** 2 cells (CIFAR-10 + CIFAR-100), ~5 h.

### C1b — does the anchor buy robustness *by itself*

Added 2026-09-02, after noticing the comparison the paper makes is not stack-matched in the direction
that matters. `ladder_p0_100ep` (anchor, $p{=}0$, no WA, no AWP) is 61.21 / 25.24 and
`at_teacherinit_matched` (label CE, **with** WA and AWP) is 57.73 / 26.46, so the reported $+2.31$ AA
is measured across a difference in stack as well as loss. Whether the anchor buys robustness with no
sensitivity-matched $\varepsilon$ and no stack has never been asked.

`abl_ce_nostack` is `ladder_p0_100ep` with the loss swapped for label cross-entropy and nothing else
changed. `abl_ce_attack` is the complementary half: the anchor as the loss, a true-label CE-PGD as the
attack. Together they say whether the gain belongs to the objective, to the attack, or to the stack.

**Cost.** 2 cells, ~4 h. Queued first, ahead of everything else in the ablation run.

---

### C2 — Features, not logits

| run | clean | AA | what it isolates |
|---|---:|---:|---|
| `tausens_kd_t1` | 58.26 | 20.84 | logit target, $\tau=1$ |
| `tausens_kd_t4` | 59.39 | **24.48** | logit target, best $\tau$ |
| `tausens_kd_t16` | 57.78 | 24.00 | logit target, $\tau=16$ |
| `tausens_fd_t1` | 62.34 | 23.93 | feature anchor + head KD, $\tau=1$ |
| `tausens_fd_t16` | 62.61 | 25.61 | feature anchor + head KD, $\tau=16$ |
| **`tausens_fd_nohd`** | **62.72** | **25.88** | feature anchor, head untouched |

Complete, and it is the **isolation** the design rests on: read point fixed at $x$, only the read
quantity changes, $24.48 \to 25.88$. All six are the same base regime (50 ep, no WA, no AWP,
$\varepsilon=8/255$), so no stack component absorbs the difference.

---

### C3 — The classifier is never trained

| run | clean | AA | |
|---|---:|---:|---|
| `tausens_fd_nohd` | 62.72 | 25.88 | head untouched — best |
| head refit, adv CE / +ls0.1 / clean CE | 62.5–62.8 | 25.15–25.65 | three variants, all below |
| `featdir_champ200_headspan` | 60.30 | 28.46 | head restricted to a subspace |
| `l2_bestrecipe_freezehead` vs `l2_bestrecipe_angeps` | 62.65/28.77 vs 62.35/28.68 | | full recipe, raw design |
| `featdir_champ200_freezehead` | 60.45 | 28.63 | full recipe, **directional** design — freezing *costs* 1.72 clean here |

Complete, and the last row is worth printing rather than hiding: freezing wins on the raw design and
loses on the directional one, because a raw student keeps its feature on the teacher's scale so the
inherited head stays calibrated. That is a mechanism, and it is the reason Theorem 2 of the internal
theory notes was withdrawn.

---

### C4 — Sensitivity-matched $\varepsilon$ is allocation, not more attack

| run | clean | AA | |
|---|---:|---:|---|
| ladder $p{=}0$ vs $p{=}1$, 50 ep | 61.33→62.94 | 26.19→26.40 | +0.49 NRR, no stack |
| ladder $p{=}0$ vs $p{=}1$, 100 ep | 61.21→62.98 | 25.24→25.43 | +0.49 NRR, no stack |
| `champ_angeps_gnorm1` | 62.51 | 28.44 | $\ell_1$ as derived vs $\ell_2$ as shipped — ties |
| ~~`champ_diffrank`~~ | 61.52 | 27.95 | ⚠ **STALE REGIME, do not cite.** `student_norm: True` + `freeze_lr_epoch: 0.65` = the pre-2026-08-31 directional design. Superseded by `champ_diffrank_l2` |
| `champ_p0_l2` (uniform) | 60.42 | 28.42 | shipped recipe, `featdir_angeps_p: 0.0`. PGD 33.11, CW 30.07, NRR 38.66 |
| `champ_diffrank_l2` | 60.90 | 28.06 | shipped recipe, difficulty-permuted. PGD 32.32, CW 29.94, NRR 38.42 — **below uniform on AA, CW and NRR**; −0.80 AA vs ours. `tab:allocation` |
| $p=0.5$ (`abl_angeps_p05`) | 62.72 | 26.37 | ran 2026-09-03, 50 ep no-stack ladder |
| $p=2.0$ (`abl_angeps_p20`) | 63.33 | 26.22 | ran 2026-09-03; NRR near-flat in $p$ -> plateau, not a tuned point |

Nearly complete. Budget preservation plus the difficulty-rank control is the strong part. The $p$
sweep is the one gap and it is cheap: if NRR is flat in $p$ the rule is a plateau rather than a tuned
point, which is the honest version of "no hyperparameter".

**Cost.** 2 cells, ~4.4 h, CIFAR-100 only.

---

### C5 — The anchor **replaces** cross-entropy rather than supplementing it

⚠ **MISSING, and it is the second-most important gap.** ARREST adds its representation term to the
label loss; our §4.1 claims the deletion is what makes Prop. 1 available and removes the weight. A
referee will ask what happens at small $\lambda$, and we cannot answer.

The nearest existing evidence is indirect: `at_teacherinit_matched` (pure CE from the teacher's
weights, 57.73/26.46) and the anchor with no stack (61.21/25.24) bracket the two ends, but nothing
sits between them.

**Design.** $\mathcal{L}_{\mathrm{anchor}} + \lambda\,\mathrm{CE}(f_s(x_{\mathrm{adv}}), y)$ for
$\lambda \in \{0.1, 0.3, 1.0\}$, base regime so nothing absorbs it. **Cost.** 3 cells, ~6.6 h.

---

### C6 — The gains are not the stack

Complete: the 8-cell ladder at two schedule lengths (§5.4), plus the layer-by-layer comparison —
50 ep with weight averaging alone reaches NRR 38.46 against ADR's 38.08 with both WA and AWP over
twice the schedule. `wadec_*` (11 runs, 08-18/19) is the earlier full decomposition of WA / AWP /
$\varepsilon_{\mathrm{tr}}$ / `freeze_lr` on both designs and is where `freeze_lr` was found to be a
net loss for the raw target; it belongs in the appendix.

---

### C7 — What transfers is geometry, not accuracy

Complete: the 5-teacher ladder (`tladder_clean_*`, 08-27) with $r=-0.999$, plus the Tiny-ImageNet
reproduction where teacher accuracy is controlled to $0.32$.

---

### C8 — Initialization is not the whole story

| run | clean | AA | |
|---|---:|---:|---|
| `at_teacherinit_matched` C100 | 57.73 | 26.46 | teacher init, label CE, no anchor |
| `at_teacherinit_matched` C10 | 82.23 | 50.72 | same |
| `featdir_scratch_200ep` C10 | 80.96 | 49.85 | anchor, **random** init, 200 ep |
| ⚠ **anchor, random init, shipped recipe** | | | **MISSING** |

The existing scratch cell is 200 epochs on the ReBAT schedule, so it is not comparable to the shipped
100-epoch cell. One matched run closes it. **Cost.** 1 cell, ~2.2 h.

---

## Axes that are settled and belong in an appendix, not the main ablation

| axis | runs | verdict |
|---|---:|---|
| Directional vs raw $\ell_2$ target | 7 + the 2$\times$2 `b2x2_*` block | **tie** at each design's own schedule (39.17 vs 39.29) |
| Subspace rank $k$ | 3 (`k512_lamda*`) + 4 (`rank_k*`) | $k{=}350$ buys $+0.31$ AA over $k{=}512$; a hyperparameter, not a mechanism |
| Consistency term $\lambda$ | 14 (`lamda*`) | flat: 63.75/26.06, 63.92/25.87, 63.73/26.11 across $\lambda \in \{0,4,15\}$ |
| Angular deadband `angtol` | 3 | **harmful**: AA 22.71 / 22.33 / 21.68 against 28.6 baseline |
| Norm penalty, prototype, ETF frame | `npen*`, `proto`, `etf` | no effect |
| `featdir_alpha` head routing | `champ_alpha1` 61.03/28.58 | irrelevant (28.58 vs 28.69) |

Six dead axes with numbers attached. They are worth one appendix table: a method with "nothing to
tune" is more credible when the things that were tried and did nothing are listed.

---

## Figures

| | content | source | status |
|---|---|---|---|
| **Fig. 1a** | conceptual diagram: teacher read at $x$, student at $x_{\mathrm{adv}}$, the $\varepsilon$-ball, and $L$/$F$/$O$ drawn on it | — | to draw |
| **Fig. 1b** | clean vs AA scatter with the trade-off frontier, our point marked | Tables 2–3 | plottable now |
| **Fig. 2** | toy: $b^\star$ against $\varepsilon$ in the two-feature model, showing the exact-zero threshold | Appendix A.1 numerics ($b^\star = 0.510$ at $0.5\eta$, $0$ at $\eta, 2\eta, 4\eta$) | **needs a sweep**, cheap, CPU only |
| **Fig. 3** | teacher-length ladder as a *curve* on the clean–AA plane, the way ARREST plots its $\phi$ points | Table 12 | plottable now |
| **Fig. 4** | local linearity: remainder proportion, natural teacher vs adversarially trained, as IGDM's Fig. 2 | Table 8 | plottable now |

Fig. 2 is the one that needs new computation and it is the cheapest thing on this page: the model is
$d=512$ Gaussians, no network. A denser $\varepsilon$ grid turns the four numbers in Appendix A.1 into
a curve with a visible kink, which is what makes the $\ell_1$-penalty claim legible.

---

## Priority

| | what | cost | why |
|---|---|---:|---|
| 1 | **C1** teacher read at $x_{\mathrm{adv}}$ | 5 h | the central claim has no direct test |
| 2 | **C5** anchor $+ \lambda\,$CE | 6.6 h | "replaces rather than supplements" is asserted, not shown |
| 3 | **Fig. 2** toy sweep | minutes | turns Appendix A.1 into a figure |
| 4 | **C4** $p$ sweep | 4.4 h | makes "no hyperparameter" a measured plateau |
| 5 | **C8** matched scratch cell | 2.2 h | separates initialization from the anchor |

About 18 GPU-hours in total, behind the AWP re-run queue. Everything else the ablation section needs
is already measured.
