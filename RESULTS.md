# Experiment results

Our own runs, scored with this repo's eval code (`utils.evaluate` / `utils.evaluate_final_aa`),
alongside the two baselines we position against. Literature-wide comparisons live in
`comparison_resnet18_cifar10.md` / `comparison_resnet18_cifar100.md`; this file is **our
measurements**.

- Threat model: ℓ∞, **ε = 8/255 at evaluation** (training ε differs per row, see the config).
- **AA** = AutoAttack, `apgd-ce` + `apgd-t`, all 10 000 test images.
- **NRR** = harmonic mean(clean, AA) — ADR's own calibration metric. Computed here from the
  clean and AA columns.
- All rows are the **`last`** checkpoint, **seed 0**, unless stated. `—` = not measured.
- Every comparison reports clean next to robust: this is a trade-off surface, not one number.

---

## 1. CIFAR-100 / ResNet-18

| run | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **`featdir_champ200_angeps`** (champion) | **62.17** | 34.77 | **30.92** | 28.59 | **39.17** |
| `featdir_champ200_100ep` (prev. champion) | 60.74 | 34.94 | 30.53 | **28.69** | 38.97 |
| `featdir_champ200_trawsnorm` | 60.26 | 35.33 | 30.49 | 28.63 | 38.82 |
| `featdir_champ200_noawp_angeps` | 62.25 | 32.49 | 30.32 | 28.13 | 38.75 |
| `featdir_champ200_headspan` | 60.30 | 35.76 | 30.31 | 28.46 | 38.67 |
| `featdir_champ200_rawfeat` | 59.79 | 34.94 | 30.13 | 28.57 | 38.66 |
| `nofeat_champ200_norm` (no feature loss) | 58.92 | 35.10 | 30.03 | 28.71 | 38.61 |
| `featdir_champ200_noawp` | 60.73 | 33.32 | 29.60 | 28.08 | 38.40 |
| `featdir_champ200_tau1` (τ 16→1) | 60.26 | 31.31 | 29.93 | 27.78 | 38.03 |
| `featdir_champ200_fullraw` | 57.78 | 35.43 | 29.68 | 28.04 | 37.76 |
| `featdir_awp_100ep_eps10` | 60.04 | 34.36 | 29.31 | 27.36 | 37.59 |
| `nofeat_champ200_raw` (diverged) | 45.84 | 29.65 | 24.43 | 22.66 | 30.33 |
| **ADR + WA + AWP** (baseline) | 57.36 | 34.92 | 30.62 | 28.50 | 38.08 |

ADR's clean/PGD/CW are our **local** re-evaluation of ADR's published checkpoint; its AA is ADR's
**published** value (see `baseline.md`).

**Two rows deserve attention rather than the ranking:**

- `nofeat_champ200_norm` removes the feature loss entirely (plain KD on `z_t/τ`, everything else
  the champion) and still reaches **AA 28.71** — a tie with the champion's 28.69. The directional
  loss's net contribution over plain KD is clean **+1.82** and CW **+0.50**, *not* AA. This is the
  largest open threat to the method's story.
- The champion wins NRR on **clean**, not AA: 28.59 vs 28.69 is a loss of 0.10, inside the day's
  observed AA noise band (28.46–28.71).

### 1b. Screening runs (50ep regime, no AA)

Used to choose directions; **PGD margins here did not transfer to AA**, so treat CW as the only
useful proxy in this table.

| run | clean | PGD-20 | CW |
|---|---:|---:|---:|
| `angeps_p10` (p = 1.0) | **64.59** | 28.77 | **27.05** |
| `angeps_p05` (p = 0.5) | 64.36 | 28.45 | 26.99 |
| `fg_plain_th_sh_kl` (p = 0 base) | 62.61 | 29.16 | 26.63 |
| `n2_traw_sraw` | 62.86 | 32.36 | 27.18 |
| `n2_tnorm_snorm` | 62.53 | 29.27 | 26.61 |
| `proto_g025` (target shrink) | 62.08 | 29.27 | 26.76 |
| `etf_rotate` (ETF rotation) | 60.48 | 26.82 | 26.54 |
| `featdir_wa_sgd_lr001` (SGD switch) | 55.43 | 22.97 | 16.32 |

---

## 2. CIFAR-10 / ResNet-18

| run | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| **`featdir_champ200_angeps`** (champion) | **84.66** | 56.74 | **53.94** | 51.87 | **64.33** |
| `featdir_champ200_100ep` | 82.52 | 57.20 | 53.74 | **51.89** | 63.71 |
| `featdir_scratch_200ep` (no warm start) | 80.96 | 55.74 | 51.54 | 49.85 | 61.71 |
| **ADR + WA + AWP** (baseline) | 83.26 | — | — | 51.18 | 63.39 |
| **CURE** (baseline) | 86.76 | 54.92 | 52.48 | 49.69 | 63.19 |

The champion recipe is **identical to the CIFAR-100 one** — the two configs differ only in
`dataset` and the teacher checkpoint path. No per-dataset tuning was done; CIFAR-10 is a
transfer, not a re-tune.

`featdir_scratch_200ep` costs **−2.02 AA** against the warm-started champion, which is what
justifies initializing from the natural teacher rather than from scratch.

---

## 3. CIFAR-10 / WRN-34-10

| run | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| `wrn_champ_angeps_mixupT` (**AA running**) | **88.31** | 58.67 | **56.72** | *pending* | *pending* |
| `featdir_wrn_mixupT` (earlier, no freeze/WA) | 87.12 | 52.50 | 50.67 | — | — |
| **ADR + WA + AWP** (baseline) | 86.11 | — | — | 55.22 | 67.29 |
| **CURE** (baseline) | 87.05 | — | — | 52.10 | 65.19 |

**The bar is NRR 67.29.** At clean 88.31 that requires **AA ≥ 54.35**. The ResNet-18 CW→AA drop on
CIFAR-10 was 2.07 (53.94 → 51.87); the same drop here would land ≈ 54.6, i.e. right on the line.

Caveat to carry into any table that uses this row: it uses a **mixup teacher**, while every
ResNet-18 row above uses a plain teacher — so it moves architecture *and* teacher at once.
`wrn_champ_angeps_plainT` is queued precisely to separate the two.

Budget note: ADR's WRN rows are **200 epochs SGD**; ours is **100 epochs**. Half the schedule.

---

## 4. Baseline reference (published)

| method | arch | dataset | clean | AA | NRR |
|---|---|---|---:|---:|---:|
| ADR | ResNet-18 | CIFAR-10 | 82.41 | 50.39 | 62.54 |
| ADR + WA | ResNet-18 | CIFAR-10 | 82.59 | 50.86 | 62.95 |
| **ADR + WA + AWP** | ResNet-18 | CIFAR-10 | 83.26 | 51.18 † | 63.39 |
| ADR | WRN-34-10 | CIFAR-10 | 84.67 | 53.24 | 65.37 |
| ADR + WA | WRN-34-10 | CIFAR-10 | 82.93 | 54.13 | 65.50 |
| **ADR + WA + AWP** | WRN-34-10 | CIFAR-10 | 86.11 | 55.22 | 67.29 |
| ADR | ResNet-18 | CIFAR-100 | 56.10 | 26.87 | 36.34 |
| ADR + WA | ResNet-18 | CIFAR-100 | 58.30 | 27.54 | 37.41 |
| **ADR + WA + AWP** | ResNet-18 | CIFAR-100 | 57.36 | 28.50 | 38.08 |
| ADR | WRN-34-10 | CIFAR-100 | 59.76 | 29.36 | 39.38 |
| ADR + WA | WRN-34-10 | CIFAR-100 | 57.42 | 30.46 | 39.80 |
| **ADR + WA + AWP** | WRN-34-10 | CIFAR-100 | 62.21 | 31.63 ‡ | 41.94 |
| CURE | ResNet-18 | CIFAR-10 | 86.76 | 49.69 | 63.19 |
| CURE | WRN-34-10 | CIFAR-10 | 87.05 | 52.10 | 65.19 |

Sources: ADR rows from the ADR repo README (`/mnt/d/research/ADR/README.md`) **except the two
marked cells**; CURE ResNet-18 from the CURE paper Table 1, CURE WRN via RPAT Table 4. NRR is
recomputed here from each row's own clean and AA, so it may differ by ≤0.1 from a value quoted
elsewhere that was rounded differently.

† The ADR **paper** reports 51.18 for this cell, the **repo README** 51.24. We keep the paper value
so this file agrees with `baseline.md` and `comparison_resnet18_cifar10.md`.
‡ Likewise 31.60 (paper Table 3) vs 31.63 (README); the README value is used here.

**CURE has no CIFAR-100 WRN number** — its CIFAR-100 table (Table 2) is ResNet-18 only and reports
PGD/C&W without AA.

Architecture check: ADR's `wideresnet-34-10` (`depth=34, width=10`) and ours
(`CIFAR10/models/WideResNet.py`, `depth=34, widen_factor=10`, channels `[16,160,320,640]`) are the
same standard WRN-34-10.

---

## 5. Design axis: direction vs full raw, by regime (2026-08-18)

The one table for "does discarding the teacher's feature magnitude help?". It kept getting
re-derived from scattered logs and mislabelled in the process, so every number below was pulled
straight from `results/<ds>/ResNet18/<cell>/*.log` with the regime read off the cell's own config.

**Only two designs are on this axis**, because only these two are internally consistent:

- **direction** — normalized on both sides everywhere: direction loss, head input, and the
  eval-time `forward()` (`student_norm: True`, `featdir_rawstudent/rawteacher: False`).
- **full raw** — unnormalized everywhere, so the feature magnitude reaches the classifier and the
  inference path is raw too (`student_norm: False`, `featdir_rawstudent/rawteacher: True`).

**Partial raw is NOT a design point** — see §5d. The head KD term is *specified* identically for
both designs (target `z_t/τ`, τ = 16, same β, same detach); full raw's student logits come out
~12x sharper because `‖Φ_s‖ ≈ 12` reaches the head, and that is a **consequence of the design**,
not a knob to equalize away.

### 5a. No stack (WA and AWP off), train ε = 8/255

| ep | dataset | direction clean / AA | full raw clean / AA | AA winner |
|---|---|---|---|---|
| 50 | C100 | 62.53 / (24.76)\* | 62.86 / (25.01)\* | raw +0.25 |
| 50 | C10 | 85.79 / **48.20** | 85.75 / **48.61** | raw +0.41 |
| 100 | C100 | 61.52 / **22.90** | *running (`wadec_raw_nowa`)* | — |
| 100 | C10 | *running* | *running* | — |

\* These two AA values were measured from `_last.pkl` **before** the 2026-08-17 checkpoint fix, so
they describe the epoch-45 model, not the epoch-49 one whose clean/PGD/CW are shown. Provisional.
**C10 50ep is currently the only unqualified evidence in this regime.**

### 5b. Full stack (WA + AWP proxy + train ε 8.8/255), 100ep — the champion regime

| dataset | angeps p | direction clean / AA | full raw clean / AA | AA gap |
|---|---|---|---|---|
| C100 | 0 | **60.74** / **28.69** | 57.78 / 28.04 | direction **+0.65** |
| C100 | 1 | **62.17** / 28.59 | 60.51 / 28.59 | **tie 0.00** |
| C10 | 0 | **82.52** / **51.89** | 81.55 / 51.44 | direction **+0.45** |
| C10 | 1 | **84.66** / 51.87 | *not run* | — |

**The regime reverses the order**: raw wins AA with no stack, direction wins with the full stack.
Reproduced on both datasets at p = 0. But direction's stack-regime AA win **does not survive giving
raw the same ε allocation** — at p = 1 the C100 gap is exactly 0.00 and only clean remains
(+1.66). The C10 p=1 raw cell is the missing check on that.

### 5c. Decomposing the stack: **WA is not what flips the order** (2026-08-18)

The stack bundles four things — WA, AWP proxy, train ε 8.8/255, `freeze_lr_epoch` — and none had
ever been isolated. `scripts/run_wadec_20260817.sh` ran the WA cell of that decomposition on
CIFAR-100 (`config/CIFAR100/wadec_{dir,raw}_{nowa,wa}.yaml`, 100ep, seed 0, angeps off).

| C100 | no WA | **WA only** | full stack |
|---|---|---|---|
| direction | 61.52 / **22.90** | 61.34 / **25.55** | 60.74 / **28.69** |
| L2 (raw) | 62.40 / **24.34** | 61.58 / **26.55** | 57.78 / 28.04 |
| AA gap | L2 **+1.44** | L2 **+1.00** | direction **+0.65** |

**WA pays both designs almost the same** (+2.65 direction, +2.21 L2) and leaves L2 ahead by +1.00 —
four times the AA noise band, so this is not a close call. The reversal is bought by the *other*
three, which are worth **+3.14 AA to direction and only +1.49 to L2**, and which cost L2 −3.80 clean
against direction's −0.60.

This **refutes the 2026-08-09 mechanism hypothesis** (the direction gradient is a pure rotation, so
its trajectory is aligned with weight averaging) and the gauge-coordinate variant of it. Whatever
makes direction the better design under the stack, it is not weight averaging.

Also settled here: `wadec_raw_nowa` is the full-L2 no-stack 100ep cell that had only ever been run
as partial raw. At 62.40 / AA 24.34 it lands within 0.05 AA of the partial-raw cell it replaces, so
the no-stack L2 advantage at 100ep (**+1.44**) stands after all — the claim withdrawn in §5d is
reinstated on clean evidence.

#### The full decomposition (2026-08-18/19)

Every cell below is 100ep, seed 0, angeps off, run on the same code. `bare` = no WA, no AWP,
ε_train = ε_eval = 8/255. Gap is positive when **L2 leads** on AA.

| regime | direction clean / AA / NRR | L2 clean / AA / NRR | AA gap |
|---|---|---|---|
| bare | 61.52 / 22.90 / 33.38 | 62.40 / **24.34** / **35.02** | L2 +1.44 |
| + WA | 61.34 / 25.55 / 36.07 | 61.58 / **26.55** / **37.10** | L2 +1.00 |
| + WA + ε 8.8 | 59.96 / 26.30 / 36.56 | 60.09 / **27.00** / **37.26** | L2 +0.70 |
| + WA + ε 8.8 + AWP | **61.41** / 28.07 / **38.53** | 60.21 / 28.14 / 38.35 | **tie +0.07** |
| + WA + ε 8.8 + freeze_lr | **60.28** / **28.14** / **38.37** | 58.82 / 27.78 / 37.74 | **direction +0.36** |
| + all four (champion) | **60.62** / **28.55** / **38.82** | 57.96 / 28.13 / 37.88 | **direction +0.42** |

**The champion row is a 2026-08-18 re-run**, not the original: `featdir_champ200_100ep` (2026-08-01)
gave 60.74 / 28.69 and `featdir_champ200_fullraw` (2026-08-02) gave 57.78 / 28.04, and the code has
changed since (angeps, the adaptive-pooling switch, the design-axis commit). All four values
reproduce inside the 0.25 AA noise band, so those runs were sound and the chain is self-consistent.

**What each element buys (AA):**

| element | direction | L2 | gap shift |
|---|---:|---:|---:|
| WA | +2.65 | +2.21 | 0.44 |
| train ε 8.8 | +0.75 | +0.45 | 0.30 |
| AWP (no freeze_lr) | +1.77 | +1.14 | 0.63 |
| freeze_lr (no AWP) | **+1.84** | +0.78 | **1.06** |
| AWP given freeze_lr | +0.41 | +0.35 | 0.06 |

Three things follow, and they pull in different directions:

1. **The design ordering is regime-dependent, and the bare regime favours L2 on both axes.** Any
   claim that the directional target is intrinsically better is refutable by row 1 of this table.
2. **But the "AA tie, clean gain" signature is not fragile.** It appears with AWP alone
   (clean +1.20 at AA −0.07), with freeze_lr alone (+1.46 at +0.36), and with both (+2.66 at
   +0.42) — three recipes, two independent elements. Only the bare and WA-only regimes favour L2,
   and no competitive AT method publishes there (ADR itself carries WA + AWP).
3. **AWP and freeze_lr are substitutes, not additive.** AWP's gap contribution collapses from 0.63
   to 0.06 once freeze_lr is present. Each alone does the job; together they add little.

The open mechanism question is why every anti-overfitting element pays the directional design more.
The leading account is that direction simply overfits more to begin with: from 50ep to 100ep in the
bare regime it loses **1.86** AA against L2's **0.67**, so it starts from a larger deficit and has
more for a regularizer to recover — the stack does not reveal a latent advantage so much as repair a
larger one, overshooting slightly into a small net win. `theory_v1.md` T.6 owns this axis.

### 5d-bis. `reformation: True` in the raw configs is a dead flag (verified 2026-08-22)

Almost every raw cell — `l2_bestrecipe_angeps`, `featdir_champ200_fullraw{,_angeps}`, all
`wadec_raw_*`, `n2_traw_sraw`, `fg_nofeat_raw_scaled`, `nofeat_champ200_raw` — carries
`reformation: True`, which reads as "normalization on" and has caused the raw/normalized confusion
more than once. It does nothing: `reformation` is consulted **only when `student_norm` is absent**
(`utils.py:459`, `methods.py:2149`), and every one of these configs sets `student_norm: False`
explicitly.

Checked by construction rather than by reading: each config was passed through `get_model`, a batch
was pushed through the resulting student, and the logits compared against `linear(scale·Φ)` and
`linear(scale·normalize(Φ))`. **All nine matched the raw form to within 1e-4**; none took the
normalized path. The architecture differs too — raw cells instantiate
`CIFAR10.models.resnet`, normalized ones `CIFAR10.models.resnet_z` — and so does the logit norm
(11.26 vs 1.30 on the same input). The training loss confirms it independently: `dir_loss_adv` is
**57.0** on the raw cell against **0.41** on the normalized one, and `2−2cos θ ≤ 4` cannot reach 57.

So every "L2 / raw" number in this file is genuinely raw. New raw configs are written with
`reformation: False` for legibility; the already-run ones are left alone so their logs stay
reproducible.

### 5d. Partial raw — diagnostic hybrid, not a design

`featdir_rawstudent/rawteacher: True` **with** `student_norm: True`: the feature loss chases the
raw teacher magnitude while the head and `forward()` both normalize it away. Training says the
magnitude matters and inference discards it, so no one would propose this as a method — it is kept
only as a diagnostic, and it must never be used as "the raw cell" in a design comparison.

| regime | cell | clean | AA |
|---|---|---:|---:|
| full stack 100ep | `featdir_champ200_rawfeat` | 59.79 | 28.57 |
| no stack 100ep | `b2x2_sraw_traw` | 62.54 | 24.29 |
| no stack 50ep | `c10_tr_sr_kl` (C10) | 86.00 | 48.73 |

⚠ This is the **third** time these cells have been mistaken for full raw (`fg_*` grid 2026-08-01,
the 2026-08-09 grid, and a 2026-08-18 clean-gap decomposition built on the first row above and then
withdrawn). Two claims that rested on them are **retracted**: "the no-stack gap grows to +1.39 at
100ep" (row 2 is partial raw; full raw at 100ep is unmeasured) and the "−0.95 feature-loss /
−2.01 magnitude" split of the champion-regime clean gap.

### 5e. Each design at its OWN best recipe — the advantage is a tie (2026-08-21)

Every direction-vs-L2 number in §5a–c puts **both** designs on the champion recipe, which was
developed on the directional design. One of its four elements, `freeze_lr_epoch`, is a net loss for
L2 and a net gain for direction, so L2 had always been compared while carrying a knob that only
costs it. Removing it and giving L2 angeps — the recipe L2 actually prefers — closes the gap:

| C100 / ResNet18 / 100ep / seed 0, angeps p=1 | clean | PGD-20 | CW | AA | NRR |
|---|---:|---:|---:|---:|---:|
| direction + freeze_lr (**champion**) | 62.17 | 34.77 | **30.92** | 28.59 | 39.17 |
| L2 + freeze_lr (champion recipe imposed) | 60.51 | 35.75 | 30.21 | 28.59 | 38.83 |
| **L2 − freeze_lr (its own best)** | **62.35** | **36.26** | 30.65 | **28.68** | **39.29** |

**All three margins between the champion and L2's own best are inside the noise band** (clean +0.18,
AA +0.09, NRR +0.12 in L2's favour). This is a tie, not an L2 win — but it is decidedly not a
direction win. `freeze_lr` alone had been costing L2 **NRR 0.46** (38.83 → 39.29), and that was the
entire gap the champion recipe used to display.

Consequences, stated plainly:

- **The headline SOTA comparison is unaffected.** Both cells beat ADR + WA + AWP (57.36 / 28.50 /
  38.08) on every axis. What changes is the *attribution*, not the numbers.
- **The design claim does not survive.** "The directional target beats the raw-L2 target" holds only
  when L2 is made to run on a schedule tuned for direction. Nothing in §5a–c should be quoted as
  evidence for the design axis without this caveat.
- Incidentally, `l2_bestrecipe_angeps` at NRR **39.29** is now the project's best CIFAR-100 result,
  and it is not the directional design.

**One corner still open.** "Each at its own best" is verified for L2 but only half-verified for
direction: freeze_lr was shown to help direction at p=0 (+0.48 AA), never at p=1.
`dir_nofreeze_angeps` (running 2026-08-21) fills that corner. If direction also prefers no
freeze_lr, the champion is not direction's optimum either and this pair must be re-formed.

### 5f. Dead axis: angular tolerance (`featdir_angtol`, 2026-08-19)

A deadband on the direction loss — samples whose angle is already inside tolerance are released
(`dir_loss *= (cos < m)`, loss shape otherwise untouched, attack unmasked). Motivated by the
observation that the no-stack run is *better* aligned than the champion and much less robust, i.e.
that over-alignment to a non-robust anchor might be the cost, and that a brake for it might belong
inside the loss rather than in the schedule.

| bare regime, 100ep | released | clean | CW | AA | final cos |
|---|---:|---:|---:|---:|---:|
| no deadband | 0% | 61.52 | 24.48 | **22.90** | 0.841 |
| `angtol 0.95` | 15% | 62.05 | 24.51 | 22.71 | 0.843 |
| `angtol 0.90` | 45% | 61.40 | 24.09 | 22.33 | 0.840 |
| `angtol 0.85` | 59% | 61.48 | 23.58 | **21.68** | 0.827 |

Monotone in the released fraction and never better than the baseline, with clean flat throughout —
a pure robustness loss, not a trade-off move. The mechanism failed for a readable reason: **the
alignment never dropped** (0.827–0.843 against the champion's 0.750 target), because releasing a
sample only zeroes *its* gradient while the shared backbone keeps aligning it anyway. Per-sample
loss gating is "discard training signal", not "align less". Also worth recording: the released
samples are the ones the attack failed to rotate, and dropping them *costs* AA — those samples were
carrying robustness.

### 5g. Head probe: how much of the gap was the head, not the representation (2026-08-20)

The backbone is clean — `featdir_alpha` defaults to 0 (`methods.py:2106`), so the head KD term is
fully detached and never reaches it. The head is not: it is fitted against `z_t/τ` while reading a
feature of norm ~1 (direction) or ~12 (raw L2), so the two designs' heads were fitted under
different conditions and the final accuracy mixes representation quality with head fit.

`scripts/head_probe.py` freezes the backbone (eval mode, BN stats frozen), re-initializes the linear
head and refits it identically for every cell: adversarial CE, 30 epochs, lr 0.1, wd 0, with the
head **input** divided by its own mean norm so both designs face the same optimization problem.
Each design keeps its own inference geometry, so this is not the partial-raw hybrid.

| cell | trained head clean / AA | refit head clean / AA |
|---|---|---|
| bare direction | 61.52 / 22.90 | 61.36 / 22.76 |
| bare L2 | 62.40 / **24.34** | 62.46 / **24.41** |
| champion direction | 60.62 / 28.55 | 60.06 / 27.96 |
| champion L2 | 57.96 / 28.13 | 58.69 / 27.81 |

The refit reproduces every trained-head number within 0.73 clean / 0.59 AA, which is the protocol's
own sanity check. Two readings:

- **Bare: unchanged.** L2 wins both axes under all three protocols. That deficit is in the
  representation, not the head.
- **Champion: the gap halves.** clean +2.66 → **+1.37**, AA +0.42 → **+0.15** (a literal tie).
  Roughly half the champion-regime clean gap and most of the AA gap were the head fit — the raw
  head had been paying for a ~12× sharpness mismatch against `z_t/16`, and a fair refit recovers
  +0.73 clean for it.

⚠ Not yet run on the **angeps** pair, which is the headline. If the same proportion holds there, the
+1.66 clean of §5e's first two rows would shrink toward ~0.9.

**A v1 of this probe is void**: it gave every cell the same lr and weight decay, which is not the
same problem for a unit-norm-input head, and the directional probes plateaued near-uniform (CE 3.48
/ 4.00 against ln 100 = 4.61) while *beating* the raw cells on train accuracy. Its champion-pair
"reversal" (direction 52.86) was an undertrained head, not a finding.

---

## 5h. Feature anchor vs logit anchor, and what the head term is worth (2026-08-23)

Base regime throughout: raw features, 50 epochs, no WA, no AWP, ε = 8/255, no interventions of any
kind. NRR is the harmonic mean of clean and AA.

| | clean | PGD-20 | CW | **AA** | **NRR** |
|---|---:|---:|---:|---:|---:|
| **A. logit anchor (pure KD)** | | | | | |
| τ = 1 | 58.26 | 22.62 | 22.79 | 20.84 | 30.70 |
| **τ = 4** | 59.39 | 30.30 | 26.84 | **24.48** | **34.67** |
| τ = 16 | 57.78 | 31.57 | 26.11 | 24.00 | 33.91 |
| **B. feature anchor + head KD** | | | | | |
| head KD τ = 1 | 62.34 | 26.34 | 26.11 | 23.93 | 34.58 |
| head KD τ = 16 | 62.61 | **32.47** | 27.32 | 25.61 | 36.35 |
| **C. feature anchor, no head KD** | | | | | |
| **head left at the teacher's** | 62.72 | 28.69 | **27.80** | **25.88** | **36.64** |
| ‥ head retrained, adversarial CE | 62.77 | 29.93 | 27.66 | 25.65 | 36.42 |
| ‥ head retrained, adv CE + label smoothing 0.1 | 62.57 | 30.43 | 27.55 | 25.66 | 36.39 |
| ‥ head retrained, adv CE + label smoothing 0.3 | 62.49 | 31.43 | 27.47 | 25.68 | 36.40 |
| ‥ head retrained, clean CE | 62.70 | 28.90 | 27.23 | 25.15 | 35.90 |

**Four readings.**

1. **The feature anchor beats the best temperature on every axis**: C's best against A's best is
   clean +3.33, CW +0.96, AA +1.40, NRR **+1.97** — and it has no temperature to have chosen.
2. **τ is squeezed from both sides.** At τ = 1 the teacher's distribution still sits at 0.820
   max-prob against the student's 0.016, so the student is asked for a confidence 20× sharper than
   its own. At τ = 16 the target is at 0.0196 max-prob, within 1% of uniform (0.0100) — nothing left
   to learn — *and* the warm-started student begins 16-fold away from it, collapsing to clean **1.01**
   in the first epoch and still recovering at the end. Best available is τ = 4 and it is not enough.
3. **The head-KD term is worth removing, not just tuning.** No temperature beats deleting it:
   36.35 at the best τ, 34.58 at a bad one, **36.64 with the term gone**.
4. **The head is best left untouched.** Every re-training we tried lands below it, and label
   smoothing lands exactly where hard labels do (36.39 / 36.40 against 36.42), so "hard labels are
   too sharp for this stage" is not the explanation.

**PGD would have inverted every one of these.** The head-KD τ = 16 row has the best PGD-20 (32.47)
and a worse AA than the row with no head term at all (25.61 vs 25.88); across the label-smoothing
sweep PGD rises monotonically (29.93 → 30.43 → 31.43) while CW falls (27.66 → 27.55 → 27.47). This
table is the cleanest instance in the project of the standing rule that PGD does not arbitrate.

**Consequence for the recipe.** β, τ and the head-distillation term are all deleted; the head is the
teacher's, untouched; the teacher appears in exactly one place, the feature target. There is no
hyperparameter left on this axis to choose.

*Teacher confidence, for reference* (clean test): natural teacher max-prob 0.820, ‖logits‖ 16.63,
entropy 0.76, clean 77.38; a trained robust student 0.016, 0.81, 4.60, 61.84. Applying τ to the
teacher gives max-prob 0.8212 / 0.1527 / 0.0196 at τ = 1 / 4 / 16, and from τ ≈ 16 upward this
tracks the first-order form 1/K + (max z − z̄)/(Kτ) with max z − z̄ = 10.35.

---

## 6. Standing caveats

1. **Seed 0 only**, every row. The clean gains (+1.4~2.1) are large, but the "AA is a tie" claim
   needs at least one more seed to be safe.
2. The angeps ablation **mixes schedules**: the base row is 50ep while the +WA rows are 100ep.
3. **CW does not predict AA.** The 50ep raw/raw cell and AWP both won on PGD/CW at 50ep and then
   failed to carry into the champion recipe. Only AA arbitrates.
4. AA is the arbiter of the headline claim, and on AA we are **tied, not ahead** — see §1.
5. **§5's AA margins are the size of the noise band.** Direction's stack-regime AA win is +0.65
   (C100) and +0.45 (C10) against an observed AA noise band of 0.25 (28.46–28.71), on one seed. The
   *clean* margins (+2.96 / +0.97) are comfortably outside it; the AA ordering is not. Read §5b's
   "direction wins" as clean-solid, AA-suggestive.
6. **Any AA measured from a `_last.pkl` saved before 2026-08-17 describes a stale model.**
   `main.py` saved the checkpoint before refreshing it from `exp_avg`, so `_last.pkl` held the
   weights from the last *interval* epoch — up to `interval - 1` epochs early (epoch 45 of a 50ep
   run at `interval: 5`). AA logged **in-run** as `last_aa_acc` is unaffected and correct; AA
   re-measured later by a script from those checkpoints is not. The true final weights of those runs
   were never saved and cannot be recovered — only re-running fixes an affected row.
