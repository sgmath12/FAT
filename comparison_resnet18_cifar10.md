# ResNet-18 / CIFAR-10 robustness comparison (by method)

Common metrics pulled from the comparison tables of the papers in this folder.
Primary threat model: **ℓ∞, ε = 8/255** (the standard AT setting; matches our workspace runs).

Metrics: **Clean** (benign test acc), **PGD-20**, **AA** (AutoAttack, standard), **C&W** where reported.

> Caveats before cross-comparing numbers:
> - The two papers are **independent reproductions** — AT/PGD-AT, TRADES, MART appear in both with different numbers (different code, schedules, checkpoint selection). Compare *within* a source, not naively across.
> - RPAT (Table 1) selects the **"best" checkpoint by PGD-20**; CURE (Table 1) reports its best-config result. Both ℓ∞, ε=8/255.
> - "—" = not reported in that paper.

---

## 1. ResNet-18, CIFAR-10, ℓ∞ — all methods, one table

One row per method (deduplicated). For methods reported in both papers (AT/TRADES/MART)
the **CURE** numbers are used (its table also has AA + C&W); RPAT-only methods are added below.

**NRR** = 2·Clean·AA/(Clean+AA), computed here from the Clean & AA columns (matches RPAT's
reported NRR formula; differs from CURE paper's NRR which uses C&W instead of AA).

| Method | Clean | PGD-20 | AA | C&W | NRR | Src |
|---|---:|---:|---:|---:|---:|:--:|
| AT (Madry) | 82.78 | 51.30 | 44.63 | 49.72 | 57.99 | C |
| TRADES | 82.41 | 52.76 | 48.37 | 50.43 | 60.96 | C |
| MART | 80.70 | 54.02 | 47.49 | 49.35 | 59.79 | C |
| FAT | 87.72 | 46.69 | 43.14 | 49.66 | 57.84 | C |
| ST-AT | 83.10 | 54.62 | 50.50 | 51.43 | 62.82 | C |
| ACT | 84.33 | 55.83 | — | — | — | C |
| ARD | 82.84 | 51.41 | — | — | — | C |
| IAD | 80.63 | 53.84 | 50.17 | 51.60 | 61.85 | C |
| LAS-AT | 82.39 | 53.70 | 49.94 | 51.96 | 62.19 | C |
| Consistency-AT | 83.42 | 51.96 | 47.72 | — | 60.71 | R |
| PGD-AT + RPAT | 83.20 | 51.29 | 48.00 | — | 60.88 | R |
| TRADES + RPAT | 80.02 | 51.21 | 47.47 | — | 59.59 | R |
| MART + RPAT | 75.44 | 52.88 | 47.03 | — | 57.94 | R |
| Consistency-AT + RPAT | 84.12 | 52.33 | 48.98 | — | 61.91 | R |
| **CURE** | **86.76** | **54.92** | 49.69 | **52.48** | **63.19** | C |
| ADR (AT+ADR) | 82.41 | — | 50.38 | — | 62.53 | A |
| F²AT | 80.56 | 50.65 | 46.54 | 47.81 | 59.00 | F |
| Generalist++ (D, NT+ℓ∞) | 89.09 | 50.01 | 46.07 | — | 60.73 | G |
| B-MTARD | 88.20 | 51.68* | 47.44 | 49.88 | 61.70 | M |
| ARREST | 86.63 | — | 46.14 | — | 60.21 | AR |

`Src`: C = CURE.pdf · R = RPAT.pdf (best-by-PGD20) · A = ADR.pdf · F = F2AT.pdf ·
G = Generalist++.pdf · M = B-MTARD.pdf · AR = ARREST.pdf. AT ≡ PGD-AT ≡ SAT.
**Each Src is an independent reproduction / setup** — compare *within* a source, not naively across
(all ℓ∞, ε=8/255, but checkpoint-selection, PGD variant, and baselines differ).
Per-row notes:
- **ADR**: paper tabulates only Clean + AA (PGD-10 used for checkpoint selection, no PGD-20 column). Its strongest ResNet-18 config AT+WA+AWP+ADR reaches **83.26 / 51.18 AA**.
- **F²AT**: proposed-method row of its Table I (own AT/TRADES/MART reproduction differs from CURE's).
- **Generalist++**: the trade-off-oriented variant Generalist-D (NT+ℓ∞); a multi-norm variant (ℓ∞+ℓ2) gives 86.94 / 50.46 / 46.24.
- **B-MTARD**: `*` PGD shown is **PGDsat** (CE-based); it also reports **PGDtrades 54.40**. Distillation method (ResNet-18 student, robust teacher WRN-34-10).

**Takeaways (ResNet-18 / CIFAR-10 / ℓ∞):**
- Highest NRR (clean·AA balance): **CURE 63.19**, **ADR 62.53**, **ST-AT 62.82**, **B-MTARD 61.70**, **Generalist++ 60.73**.
- Highest clean: **Generalist++ 89.09** and **B-MTARD 88.20** (distillation/multi-learner), then FAT 87.72 — but all three trail on AA.
- Highest AA: **ST-AT 50.50**, **ADR 50.38**, **CURE 49.69** — the strongest robustness with still-competitive clean.
- RPAT lifts AA over each of its base methods (e.g. Consistency-AT 47.72 → 48.98) at ~flat clean.

---

## 2. Same comparison, ℓ2 norm (RPAT Table 2, ResNet-18, CIFAR-10) — reference
| Method | Clean | PGD-20 | AA |
|---|---:|---:|---:|
| PGD-AT | 87.76 | 67.92 | 66.36 |
| TRADES | 83.99 | 68.60 | 65.93 |
| MART | 84.09 | 68.32 | 66.28 |
| Consistency-AT | 88.76 | 69.35 | 67.46 |
| PGD-AT + RPAT | 88.20 | 68.56 | 67.63 |
| TRADES + RPAT | 85.17 | 68.95 | 67.67 |
| MART + RPAT | 84.65 | 68.60 | 66.69 |
| Consistency-AT + RPAT | 89.38 | 70.41 | **69.44** |

---

## 3. SOTA trade-off tables (NOT ResNet-18) — for context only

RPAT's headline SOTA comparison uses **PreActResNet-18** and **WideResNet-34-10**, not ResNet-18.
Useful because **CURE also appears here** (WRN-34-10), giving a cross-paper anchor.

### PreActResNet-18, CIFAR-10, ℓ∞ (RPAT Table 3)
| Method | Clean | AA |
|---|---:|---:|
| WA | 83.50 | 49.89 |
| MMA | 85.50 | 37.20 |
| AWP | 81.11 | 50.09 |
| GAIRAT | 78.70 | 37.70 |
| KD+SWA | 84.06 | 49.82 |
| EWAT | 82.80 | 48.20 |
| MAIL | 79.50 | 39.60 |
| TE | 82.04 | 50.12 |
| SOVR | 81.90 | 49.40 |
| ReBAT | 82.09 | 50.72 |
| **RPAT++** | 82.63 | **51.00** |

### WideResNet-34-10, CIFAR-10, ℓ∞ (RPAT Table 4)
| Method | Clean | AA |
|---|---:|---:|
| WA | 87.66 | 52.65 |
| MMA | 87.80 | 43.10 |
| AWP | 85.63 | 53.32 |
| GAIRAT | 83.00 | 41.80 |
| KD+SWA | 87.45 | 53.59 |
| EWAT | 86.00 | 51.60 |
| MAIL | 82.20 | 43.30 |
| TE | 85.97 | 52.88 |
| SOVR | 85.00 | 53.10 |
| ReBAT | 85.25 | 54.78 |
| ADR | 84.67 | 53.25 |
| CURE | 87.05 | 52.10 |
| **RPAT++** | 86.76 | **54.97** |
| ReBAT* | 86.66 | 55.64 |
| **RPAT++*** | 87.57 | **55.79** |

(* = with the optional ReBAT trick.)

---

## Sources (PDFs in this folder)
- **CURE.pdf** — "Conserve-Update-Revise to Cure Generalization and Robustness Trade-off in Adversarial Training", ICLR 2024 (arXiv 2401.14948). Table 1 (CIFAR-10, ResNet-18 + WRN-34-10).
- **RPAT.pdf** — "Failure Cases Are Better Learned But Boundary Says Sorry: …Accuracy-Robustness Trade-Off" (RPAT / RPAT++), ICCV 2025 (arXiv 2508.02186). Table 1–2 (ResNet-18, ℓ∞/ℓ2), Table 3 (PreActResNet-18), Table 4 (WRN-34-10).
- **ADR.pdf** — "Annealing Self-Distillation Rectification Improves Adversarial Training", ICLR 2024 (arXiv 2305.12118). Table 1–2 (ResNet-18 / WRN-34-10).
- **F2AT.pdf** — "F²AT: Feature-Focusing Adversarial Training via Disentanglement of Natural and Perturbed Patterns" (arXiv 2310.14561). Table I (ResNet-18, CIFAR-10, white-box).
- **Generalist++.pdf** — "Generalist++: A Meta-learning Framework for Mitigating Trade-off in Adversarial Training" (arXiv 2510.13361). Table I (ResNet-18 / WRN-32-10, CIFAR-10).
- **B-MTARD.pdf** — "Mitigating Accuracy-Robustness Trade-off via Balanced Multi-Teacher Adversarial Distillation", IEEE TPAMI 2024 (arXiv 2306.16170). Table 3 (ResNet-18 student, CIFAR-10).
