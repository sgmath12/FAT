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

\* AA for ADR is the **published** number (paper/checkpoint README claims AA=28.52,
clean=57.36 — matches our locally-reproduced clean=57.37 almost exactly, so the checkpoint/repro
setup is trusted); AutoAttack was not re-run locally for this checkpoint yet (clean/FGSM/PGD/CW
above ARE from the local run, `scratchpad/adr_eval_result.txt`).

**Derived metrics** (H(x) = harmonic mean(clean, x); NRR = harmonic mean(clean, AA), ADR's own
calibration metric):

| | H(pgd20) | H(cw) | NRR |
|---|----------|-------|-----|
| ADR (WA+AWP) | 43.42 | 39.93 | 38.08 (published) |
| Ours (champion) | 44.07 | 39.11 | 37.06 |
| **Ours (100ep+AWP, train_eps 10)** | **43.71** | **39.39** | **37.59** |

Reading: ours trades clean accuracy up (+5.4) for a bit of robustness down (PGD H roughly
ties, CW H −0.8, NRR −1.0 vs ADR using ADR's own published AA). See [`README.md`](README.md)
for the champion's config/loss/repro command.

**Caveat on the trade-off framing (2026-07-30):** at *matched clean* we lose outright. Our
robust-leaning eps10 variant is 58.32 clean / AA 26.80 against ADR-full's 57.36 / 28.50 — a
**−1.70 AA** deficit at the same clean accuracy. The like-for-like comparison that we do win is
against `AT+ADR` (56.10 / AA 26.87 / NRR 36.34, no WA+AWP stack): NRR +0.72 at +6.65 clean.
The entire remaining deficit is AWP, which contributes +1.63 AA in ADR's own ablation —
see [`awp_memory.md`](awp_memory.md) for our AWP port and why it has not closed the gap yet.
