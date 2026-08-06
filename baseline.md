# Baseline: ADR (ICLR'24)

The #1 baseline this project compares against. ADR's own published CIFAR100/ResNet18
checkpoint (`resnet18_cifar100_adr_wa_awp.pt`, WA/EMA weights) was loaded into ADR's own
model class (`model.resnet.ResNet`, standard single-ReLU BasicBlock — **not** this repo's
double-ReLU `resnet_z`, so the architecture exactly matches how the checkpoint was trained)
and scored with FAT's own eval code (`utils.evaluate` / `utils.evaluate_final_aa`), so the
clean/pgd/cw numbers below are directly comparable to this repo's results.

Reproduce: `scripts/eval_adr_checkpoint.py <ckpt> --ema` (see `scripts/run_adr_verify.sh` for
the download + extract + verify pipeline).

## CIFAR100, ResNet18

| | clean | FGSM | PGD-20 | PGD-10 | PGD-50 | CW | AutoAttack |
|---|-------|------|--------|--------|--------|----|------------|
| **ADR (WA+AWP)** | 57.37 | 36.90 | 34.92 | 35.26 | 34.76 | 30.62 | 28.52* |
| **Ours (champion, k350+WA+lamda4)** | 62.75 | 36.48 | 33.96 | 34.18 | 33.93 | 28.41 | 26.29 |
| Ours, 100ep, no AWP (control) | 62.04 | 35.82 | 32.38 | 32.78 | 32.44 | 27.23 | 25.16 |
| Ours, 100ep + AWP (proxy, g0.005) | 63.07 | 36.79 | 34.12 | 34.59 | 34.09 | 28.20 | 25.98 |
| Ours, k512 (subspace mechanism OFF) | 63.23 | 35.54 | 32.85 | 33.16 | 32.67 | 28.43 | 25.98 |
| **Ours, 100ep+AWP, train_eps 10/255** | 60.04 | 36.56 | 34.36 | 34.59 | 34.37 | **29.31** | **27.36** |
| **Ours, `featdir_champ200_100ep` (2026-08-01)** | **60.74** | **37.44** | **34.94** | **35.24** | **34.86** | **30.53** | **28.69** |
| **Ours, `featdir_champ200_angeps` (2026-08-04)** | **62.17** | 37.24 | 34.77 | 35.09 | 34.77 | **30.92** | 28.59 |

\* AA for ADR is the **published** number (paper/checkpoint README claims AA=28.52,
clean=57.36 — matches our locally-reproduced clean=57.37 almost exactly, so the checkpoint/repro
setup is trusted); AutoAttack was not re-run locally for this checkpoint yet (clean/FGSM/PGD/CW
above ARE from the local run, `scratchpad/adr_eval_result.txt`).

**Derived metrics** (H(x) = harmonic mean(clean, x); NRR = harmonic mean(clean, AA), ADR's own
calibration metric):

| | H(pgd20) | H(cw) | NRR |
|---|----------|-------|-----|
| ADR (WA+AWP) | 43.42 | 39.93 | 38.08 (published) |
| Ours (50ep champion) | 44.07 | 39.11 | 37.06 |
| Ours (100ep+AWP, train_eps 10) | 43.71 | 39.39 | 37.59 |
| **Ours (`featdir_champ200_100ep`, 2026-08-01)** | **44.36** | **40.64** | **38.97** |
| **Ours (`featdir_champ200_angeps`, 2026-08-04)** | **44.60** | **41.30** | **39.17** |

**Reading (updated 2026-08-04): the angular budget adds clean at matched AA.**
`featdir_champ200_angeps` (§3.2a of [`METHOD.md`](METHOD.md)) reallocates the *same total*
$\varepsilon$ budget across the batch so that the **angular** displacement is equalized rather than
the pixel radius. Against the previous champion it gains clean **+1.43** and CW **+0.39** for
AA **−0.10** — a tie, inside the day's observed AA noise band (28.46–28.71). NRR 39.17 is the
project record and leads ADR-full by **+1.09**.

State this as *"recovers natural accuracy at matched robustness"*; **do not claim an AA
improvement.** The same signature reproduces on CIFAR-10 with no re-tuning (84.66 / CW 53.94 /
AA 51.87 / NRR 64.33 vs the $p{=}0$ champion's 82.52 / 53.74 / 51.89 / 63.71): AA tied, clean
+2.14. Both are seed 0 only.

**Reading (2026-08-01): the trade-off is gone.** `featdir_champ200_100ep` beats
ADR-full on **AA (28.69 vs 28.50)** *and* on clean (**+3.38**), with PGD-20 (+0.02) and CW
(−0.09) tied. All three derived metrics are ahead: H(pgd) +0.94, H(cw) +0.71, NRR +0.89. The
earlier framing below ("we trade clean up for robustness down") described the 50ep/eps10
configurations and is retained for history.

The recipe that did it: 200ep plain teacher, 100 epochs, `freeze_lr_epoch 0.65`, `wa_start 0.2`,
train_eps 8.8/255, lamda 0, k512, AWP proxy g0.005/warmup10. Full description and the open
attribution question (five levers moved at once) in [`METHOD.md`](METHOD.md) §7.

Historical reading: ours trades clean accuracy up (+5.4) for a bit of robustness down (PGD H roughly
ties, CW H −0.8, NRR −1.0 vs ADR using ADR's own published AA). See [`README.md`](README.md)
for the champion's config/loss/repro command.

**Caveat on the trade-off framing (2026-07-30, SUPERSEDED 2026-08-01 — kept for history; the
new champion leads on both axes so the matched-clean argument below no longer applies):** at
*matched clean* we lose outright. Our
robust-leaning eps10 variant is 58.32 clean / AA 26.80 against ADR-full's 57.36 / 28.50 — a
**−1.70 AA** deficit at the same clean accuracy. The like-for-like comparison that we do win is
against `AT+ADR` (56.10 / AA 26.87 / NRR 36.34, no WA+AWP stack): NRR +0.72 at +6.65 clean.
The entire remaining deficit is AWP, which contributes +1.63 AA in ADR's own ablation —
see [`awp_memory.md`](awp_memory.md) for our AWP port and why it has not closed the gap yet.
