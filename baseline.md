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
| **Ours (champion, k350+WA+lamda4)** | 62.75 | — | 33.96 | 34.18 | — | 28.41 | 26.29 |

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

Reading: ours trades clean accuracy up (+5.4) for a bit of robustness down (PGD H roughly
ties, CW H −0.8, NRR −1.0 vs ADR using ADR's own published AA). See [`README.md`](README.md)
for the champion's config/loss/repro command.
