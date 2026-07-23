# ResNet-18 / CIFAR-100 robustness comparison (by method)

Common metrics pulled from the comparison tables of the papers in this folder.
Primary threat model: **ℓ∞, ε = 8/255** (the standard AT setting; matches our workspace runs).

Metrics: **Clean** (benign test acc), **PGD-20**, **AA** (AutoAttack, standard), **C&W** where reported.

> Caveats before cross-comparing numbers (stronger than on CIFAR-10):
> - These are **independent reproductions** — AT/PGD-AT, TRADES, MART appear in several papers with different numbers (different code, schedules, checkpoint selection). Compare *within* a source, not naively across. The spread is wider on CIFAR-100 (e.g. reproduced PGD-AT clean ranges **51.83 → 57.27**).
> - **CURE's CIFAR-100 table (its Table 2) does not report AutoAttack** — it gives PGD-20, PGD-100, C&W only, and its NRR is computed from **C&W**. So CURE's CIFAR-100 rows live in a separate block below and are *not* on the AA axis.
> - **RPAT** (Table 1) selects the **"best" checkpoint by PGD-20**; **ADR** selects by **PGD-10** (so it has no PGD-20 column).
> - **ARREST** only used ResNet-18 on CIFAR-10; its CIFAR-100 numbers are **WRN-34-10**, so it is excluded from the ResNet-18 table (see §3).
> - "—" = not reported in that paper for ResNet-18 / CIFAR-100.

---

## 1. ResNet-18, CIFAR-100, ℓ∞ — methods that report AutoAttack

One row per method/config. **NRR** = 2·Clean·AA/(Clean+AA), computed here from the Clean & AA
columns (matches RPAT's reported NRR formula). For the shared baselines (AT/TRADES/MART/
Consistency-AT) the **RPAT** numbers are used here because they are the ones reported *with AA*.

| Method | Clean | PGD-20 | AA | C&W | NRR | Src |
|---|---:|---:|---:|---:|---:|:--:|
| PGD-AT | 56.56 | 28.80 | 25.02 | — | 34.69 | R |
| TRADES | 55.39 | 29.36 | 24.51 | — | 33.98 | R |
| MART | 49.83 | 30.38 | 25.00 | — | 33.30 | R |
| Consistency-AT | 58.53 | 29.28 | 25.39 | — | 35.42 | R |
| PGD-AT + RPAT | 58.22 | 29.16 | 24.88 | — | 34.86 | R |
| TRADES + RPAT | 57.50 | 29.42 | 25.05 | — | 34.90 | R |
| MART + RPAT | 50.72 | 30.33 | 25.34 | — | 33.80 | R |
| Consistency-AT + RPAT | 60.33 | 29.97 | 26.31 | — | 36.64 | R |
| ADR (AT+ADR) | 56.10 | — | 26.87 | — | 36.34 | A |
| **ADR (AT+WA+AWP+ADR)** | 57.36 | — | **28.50** | — | **38.08** | A |
| F²AT | 54.19 | 26.75 | 23.24 | 25.14 | 32.53 | F |
| Generalist++ (D, NT+ℓ∞) | 62.97 | 29.48 | 23.96 | — | 34.71 | G |
| Generalist++ (D, ℓ∞+ℓ2) | 60.90 | 29.43 | 24.23 | — | 34.67 | G |
| B-MTARD | 65.08 | 28.50* | 23.98 | 25.45 | 35.05 | M |
| DP-FAT (fast-AT track) | 60.40 | 33.31 | 26.63 | 28.38 | 36.96 | D |

`Src`: R = RPAT.pdf (best-by-PGD20) · A = ADR.pdf · F = F2AT.pdf ·
G = Generalist++.pdf · M = B-MTARD.pdf · D = DPFAT.pdf. AT ≡ PGD-AT ≡ SAT.
**Each Src is an independent reproduction / setup** — compare *within* a source, not naively across
(all ℓ∞, ε=8/255, but checkpoint-selection, PGD variant, and baselines differ).
Per-row notes:
- **ADR**: paper tabulates only Clean + AA (PGD-10 used for checkpoint selection, no PGD-20 column). `AT+ADR` is the plain combination; `AT+WA+AWP+ADR` is its strongest ResNet-18 config (its CIFAR-100 SOTA claim).
- **F²AT**: proposed-method row of its Table IV (its own SAT/TRADES/MART reproduction: SAT 51.83/22.55/AA 19.32, TRADES 53.46/25.98/AA 21.96, MART 51.35/26.28/AA 21.81, SEAT 53.74/26.25/AA 22.49).
- **Generalist++**: trade-off variant Generalist-D (NT+ℓ∞); the multi-norm variant (ℓ∞+ℓ2) is the second row. Numbers are AA∞ from its Table II.
- **B-MTARD**: `*` PGD shown is **PGDsat** (CE-based); it also reports **PGDtrades 29.94**. Distillation (ResNet-18 student, robust teacher WRN-70-16). Clean 65.08 is the highest in this table.
- **DP-FAT**: a **fast-AT / efficiency** paper — entirely different baseline set (Free-AT, GAT, NuAT, PGI, PGK, …), 3-step PGD training. Not directly comparable to the multi-epoch AT methods above; included because it reports ResNet-18 / CIFAR-100 with full Clean/PGD-20/C&W/AA.

**Takeaways (ResNet-18 / CIFAR-100 / ℓ∞, AA-reported subset):**
- Highest AA: **ADR (AT+WA+AWP+ADR) 28.50** (stacked with WA+AWP), then **DP-FAT 26.63** (separate fast-AT track), **Consistency-AT+RPAT 26.31**, plain **ADR (AT+ADR) 26.87**.
- Highest NRR (clean·AA balance): **ADR-full 38.08**, **DP-FAT 36.96**, **Consistency-AT+RPAT 36.64**, **ADR(AT+ADR) 36.34**, **B-MTARD 35.05**.
- Highest clean: **B-MTARD 65.08** and **Generalist++ 62.97** (distillation / multi-learner), then DP-FAT 60.40 — but all three trail the ADR-stacked config on AA.
- RPAT lifts AA over each of its base methods (e.g. Consistency-AT 25.39 → 26.31) at ~flat or higher clean, same pattern as CIFAR-10.

---

## 1b. ResNet-18, CIFAR-100, ℓ∞ — CURE block (PGD/C&W only, no AutoAttack)

CURE's CIFAR-100 results (its Table 2) report PGD-20 / PGD-100 / C&W but **no AA**, so they cannot
be merged onto the AA axis above. **NRR here is CURE's own, computed from C&W** (not comparable
to the Clean·AA NRR in §1).

| Method | Clean | PGD-20 | PGD-100 | C&W | NRR (C&W) | Src |
|---|---:|---:|---:|---:|---:|:--:|
| AT (Madry) | 57.27 | 26.66 | 26.29 | 24.89 | 34.69 | C |
| TRADES | 57.94 | 29.25 | 29.10 | 25.88 | 35.77 | C |
| MART | 55.03 | 28.25 | 28.10 | 26.60 | 35.86 | C |
| FAT | 61.61 | 18.35 | 17.98 | 19.31 | 29.40 | C |
| HAT | 58.73 | 27.92 | — | 24.60 | 34.67 | C |
| ST-AT | 58.44 | 30.53 | 30.39 | 26.70 | 36.65 | C |
| **CURE** | **60.72** | **30.81** | 29.82 | **27.95** | **38.27** | C |

`Src`: C = CURE.pdf (ResNet-18, CIFAR-100, ℓ∞, ε=8/255).
**Within this block** CURE has the best C&W (27.95) and best C&W-NRR (38.27) while keeping high
clean (60.72); FAT is the clean leader (61.61) but collapses under attack.

---

## 2. Same comparison, ℓ2 norm (RPAT Table 2, ResNet-18, CIFAR-100) — reference
| Method | Clean | PGD-20 | AA |
|---|---:|---:|---:|
| PGD-AT | 65.00 | 41.54 | 39.27 |
| TRADES | 61.25 | 43.05 | 40.15 |
| MART | 60.08 | 43.42 | 39.85 |
| Consistency-AT | 65.14 | 42.28 | 39.92 |
| PGD-AT + RPAT | 65.14 | 41.61 | 39.23 |
| TRADES + RPAT | 62.41 | 43.56 | 40.36 |
| MART + RPAT | 60.57 | 43.92 | 40.36 |
| Consistency-AT + RPAT | 65.54 | 42.81 | **40.26** |

---

## 3. CIFAR-100 trade-off tables (NOT ResNet-18) — for context only

### PreActResNet-18, CIFAR-100, ℓ∞ (RPAT Table 3)
| Method | Clean | AA |
|---|---:|---:|
| WA | 57.26 | 25.83 |
| MMA | 60.60 | 18.40 |
| AWP | 54.10 | 25.16 |
| GAIRAT | 52.00 | 19.80 |
| KD+SWA | 57.17 | 25.66 |
| EWAT | 54.20 | 23.52 |
| MAIL | 46.50 | 16.70 |
| TE | 56.41 | 25.84 |
| SOVR | 52.10 | 24.30 |
| ReBAT | 56.13 | 27.60 |
| **RPAT++** | 56.84 | **27.68** |

PreActResNet-18, CIFAR-100, ℓ2 (RPAT Table 3): ReBAT 65.58 / 42.67 · **RPAT++ 65.63 / 42.85**.
(RPAT excludes ADR and CURE from PreActResNet-18 for lack of a prior record.)

### CIFAR-100 results on larger / other backbones (ADR Table 3, F²AT Table III, Generalist++ Table II, ARREST Table 1)
| Method | Backbone | Extra data | Clean | AA |
|---|---|:--:|---:|---:|
| AT+ADR | ResNet-18 | — | 57.36 | 28.50 |
| Rebuffi et al. | PreActResNet-18 | DDPM | 56.87 | 28.50 |
| Rade & Moosavi | PreActResNet-18 | DDPM | 61.50 | 28.88 |
| AT+ADR | PreActResNet-18 | DDPM | 57.88 | 29.59 |
| Addepalli et al. | WRN-34-10 | — | 68.74 | 31.30 |
| AT+ADR | WRN-34-10 | — | 62.21 | 31.60 |
| AT+ADR | WRN-34-10 | DDPM | 59.60 | 32.19 |
| SAT | WRN-34-10 | — | 53.64 | 21.01 | 
| F²AT | WRN-34-10 | — | 60.21 | 26.91 |
| Generalist-D (NT+ℓ∞) | WRN-32-10 | — | 66.66 | 26.86 |
| Generalist-D (ℓ∞+ℓ2) | WRN-32-10 | — | 64.85 | 27.29 |
| LBGAT | WRN-34-10 | — | 70.25 | 26.73 |
| ARREST | WRN-34-10 | — | 73.05 | 24.32 |

(Mixed backbones — read the `Backbone` column; these are *not* ResNet-18. ARREST and LBGAT push
clean very high on WRN at the cost of AA; ADR+DDPM and Addepalli lead AA on WRN-34-10.)

---

## Sources (PDFs in this folder)
- **CURE.pdf** — "Conserve-Update-Revise to Cure Generalization and Robustness Trade-off in Adversarial Training", ICLR 2024 (arXiv 2401.14948). Table 2 (CIFAR-100 + SVHN, ResNet-18; PGD/C&W, no AA).
- **RPAT.pdf** — "Failure Cases Are Better Learned But Boundary Says Sorry: …Accuracy-Robustness Trade-Off" (RPAT / RPAT++), ICCV 2025 (arXiv 2508.02186). Table 1 (ResNet-18, ℓ∞, incl. CIFAR-100), Table 2 (ℓ2), Table 3 (PreActResNet-18, CIFAR-100).
- **ADR.pdf** — "Annealing Self-Distillation Rectification Improves Adversarial Training", ICLR 2024 (arXiv 2305.12118). Table 1–2 (ResNet-18 / WRN-34-10, CIFAR-100), Table 3 (CIFAR-100 SOTA comparison).
- **F2AT.pdf** — "F²AT: Feature-Focusing Adversarial Training via Disentanglement of Natural and Perturbed Patterns" (arXiv 2310.14561). Table IV (ResNet-18, CIFAR-100, white-box).
- **Generalist++.pdf** — "Generalist++: A Meta-learning Framework for Mitigating Trade-off in Adversarial Training" (arXiv 2510.13361). Table II (ResNet-18 / WRN-32-10, CIFAR-100).
- **B-MTARD.pdf** — "Mitigating Accuracy-Robustness Trade-off via Balanced Multi-Teacher Adversarial Distillation", IEEE TPAMI 2024 (arXiv 2306.16170). Table 3 (ResNet-18 student, CIFAR-100).
- **ARREST.pdf** — "Adversarial Finetuning with Latent Representation Constraint…" (ARREST). Table 1 (CIFAR-100 on **WRN-34-10**) — ResNet-18 used on CIFAR-10 only.
- **DPFAT.pdf** — "DP-FAT" fast adversarial training (ECCV 2026 submission #6070). Table 3 (ResNet-18, CIFAR-100, ℓ∞); fast-AT baselines, not directly comparable to multi-epoch AT.
</content>
</invoke>
