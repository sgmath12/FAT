# WRN-34-10, published numbers, CIFAR-10 and CIFAR-100

2026-09-03. Everything below is AutoAttack unless the row says otherwise, $\ell_\infty$, $\epsilon = 8/255$,
transcribed from the paper named in the last column. Collected because our two WideResNet tables are
empty and the other server needs to know what line it is aiming at.

**Read the source column.** The same method is reported at different numbers by different papers, and
in one case the spread is eight points — see the caveats at the bottom before quoting any single row.

## CIFAR-10, WRN-34-10

| Method | Clean | AA | NRR | Source |
|---|---:|---:|---:|---|
| AT | 85.15 | 52.12 | 64.66 | ADR Tab. 2 |
| AT | 85.17 | 44.04 | 58.06 | CURE Tab. 1 |
| AT | 87.14 | 44.04 | 58.51 | ARREST Tab. 1 |
| AT + WA | 83.48 | 53.97 | 65.56 | ADR Tab. 2 |
| **AT + WA + AWP** | **87.42** | **55.01** | **67.53** | ADR Tab. 2 |
| TRADES | 84.73 | 52.95 | 65.17 | CURE Tab. 1 |
| MART | 83.62 | 51.23 | 63.54 | CURE Tab. 1 |
| FAT | 86.60 | 47.48 | 61.33 | CURE Tab. 1 |
| ST-AT | 84.92 | 53.54 | 65.67 | CURE Tab. 1 |
| IAD | 83.06 | 52.68 | 64.47 | CURE Tab. 1 |
| LAS-AT | 85.24 | 53.58 | 65.80 | CURE Tab. 1 |
| LAS-AT | 86.23 | 53.58 | 66.09 | ARREST Tab. 1 |
| WA | 87.66 | 52.65 | 65.79 | RPAT Tab. 4 |
| MMA | 87.80 | 43.10 | 57.82 | RPAT Tab. 4 |
| AWP | 85.63 | 53.32 | 65.72 | RPAT Tab. 4 |
| AWP | 85.57 | 54.04 | 66.24 | ARREST Tab. 1 |
| GAIRAT | 83.00 | 41.80 | 55.60 | RPAT Tab. 4 |
| KD + SWA | 87.45 | 53.59 | 66.46 | RPAT Tab. 4 |
| EWAT | 86.00 | 51.60 | 64.50 | RPAT Tab. 4 |
| MAIL | 82.20 | 43.30 | 56.72 | RPAT Tab. 4 |
| TE | 85.97 | 52.88 | 65.48 | RPAT Tab. 4 |
| SOVR | 85.00 | 53.10 | 65.37 | RPAT Tab. 4 |
| S2O | 85.67 | 54.10 | 66.32 | ARREST Tab. 1 |
| ReBAT | 85.25 | 54.78 | 66.70 | RPAT Tab. 4 |
| LBGAT + TRADES ($\alpha{=}0$) | **88.22** | 52.86 | 66.11 | LBGAT Tab. 5 |
| LBGAT + TRADES ($\alpha{=}6$) | 81.98 | 53.14 | 64.48 | LBGAT Tab. 5 |
| LBGAT | 88.22 | 52.18 | 65.57 | ARREST Tab. 1 |
| **ARREST** | **90.24** | 50.20 | 64.51 | ARREST Tab. 1 |
| ADR | 84.67 | 53.25 | 65.38 | ADR Tab. 2 |
| **ADR + WA + AWP** | 86.11 | **55.26** | **67.32** | ADR Tab. 2 |
| CURE | 87.05 | 52.10 | 65.19 | CURE Tab. 1 |
| **RPAT++** | 86.76 | **54.97** | **67.30** | RPAT Tab. 4 |

Best clean **90.24** (ARREST), best AA **55.26** (ADR + WA + AWP), best NRR **67.53**
(AT + WA + AWP, ADR's own baseline).

## CIFAR-100, WRN-34-10

Far fewer papers report this cell, and the two that report the extremes do not overlap in the middle.

| Method | Clean | AA | NRR | Source |
|---|---:|---:|---:|---|
| AT | 61.12 | 28.45 | 38.83 | ADR Tab. 2 |
| AT | 59.59 | 22.86 | 33.04 | ARREST Tab. 1 |
| AT + WA | 60.04 | 30.22 | 40.20 | ADR Tab. 2 |
| **AT + WA + AWP** | **63.11** | 30.73 | 41.33 | ADR Tab. 2 |
| LAS-AT | 61.80 | 29.03 | 39.50 | ARREST Tab. 1 |
| AWP | 60.38 | 28.86 | 39.05 | ARREST Tab. 1 |
| S2O | 63.40 | 27.60 | 38.46 | ARREST Tab. 1 |
| LBGAT + TRADES ($\alpha{=}0$) | **70.03** | 27.05 | 39.03 | LBGAT Tab. 5 |
| LBGAT + TRADES ($\alpha{=}6$) | 60.43 | 29.34 | 39.50 | LBGAT Tab. 5 |
| LBGAT | 70.25 | 26.73 | 38.73 | ARREST Tab. 1 |
| **ARREST** | **73.05** | 24.32 | 36.49 | ARREST Tab. 1 |
| ADR | 59.76 | 29.35 | 39.37 | ADR Tab. 2 |
| **ADR + WA + AWP** | 62.21 | **31.60** | **41.91** | ADR Tab. 2 |
| CURE | — | — | — | *not reported* |
| RPAT++ | — | — | — | *not reported* |

Best clean **73.05** (ARREST), best AA **31.60** (ADR + WA + AWP), best NRR **41.91** (same).

**CURE has no CIFAR-100 WideResNet number at all** — its CIFAR-100 table is ResNet-18 and reports
PGD and C&W without AutoAttack. **RPAT's WideResNet table is CIFAR-10 only.** So on this cell the
comparison line is ADR, and nothing else in the trade-off literature contests it.

## What the two datasets say about where the gap is

On CIFAR-10 the frontier is crowded: eleven methods sit between 52 and 55 AA at 85–88 clean, and NRR
across all of them spans 64.4 to 67.5. Winning there means moving a decimal.

On CIFAR-100 it is not crowded, it is **bimodal**. One end raises clean and gives up robustness —
ARREST 73.05 / 24.32, LBGAT 70.03 / 27.05. The other raises robustness and gives up clean —
ADR + WA + AWP 62.21 / 31.60. **Nothing occupies the middle**, and the NRR column shows why it
matters: every row on this dataset lands between 36.5 and 41.9 regardless of which end it came from.
A method at, say, 68 / 30 would score NRR 41.63 and dominate ARREST and LBGAT outright on both axes.

That is the cell to aim at, and it is also the cell where our own claim is easiest to state, since
ARREST and LBGAT are both natural-teacher methods and both are on the clean end.

## Caveats before quoting any row

- **"AT" is reported three times on CIFAR-10 and the AA spread is eight points** — 52.12 (ADR),
  44.04 (CURE and ARREST). Same architecture, same attack, different training recipe: ADR trains 200
  epochs with a step schedule, and the others do not say. Never quote a plain-AT WideResNet row
  without its source.
- **LBGAT is quoted differently by itself and by ARREST**: 88.22 / 52.86 and 70.03 / 27.05 in its own
  Table 5, against 88.22 / 52.18 and 70.25 / 26.73 in ARREST's Table 1. We use LBGAT's own.
- **LBGAT's natural branch is a ResNet-18 even when the robust model is WRN-34-10**, per its Table 5
  caption. Its clean accuracy is therefore not a WideResNet teacher's.
- ADR's README reports 55.22 / 31.63 where its paper reports **55.26 / 31.60**. The paper is the
  citable source; the memory note quoting the README predates this check.
- ARREST reports CIFAR-100 AA of 24.32 at clean 73.05. That is the highest clean and the lowest AA in
  the table, which is the trade-off working exactly as advertised, not an error.
- NRR is computed here from the quoted clean and AA; where a paper prints its own NRR it uses C&W
  rather than AutoAttack (CURE does this), so the two do not agree and ours is the AA version.
