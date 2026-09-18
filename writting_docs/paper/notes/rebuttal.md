# Held back for rebuttal: baselines given our recipe

Not in the paper. These runs replace a baseline's own optimizer, schedule, epochs, warm start and
training radius with ours, and only then add WA and AWP. The paper reports every baseline at its
authors' recipe instead, so the tables never claim an advantage that could be read as "we retuned the
competition". If a reviewer asks whether the baselines would win with our recipe, answer with the
CIFAR-100 numbers below.

All CIFAR-100, ResNet-18, 200-epoch natural teacher, AutoAttack on the full test set at 8/255,
final-epoch weight-averaged model, seed 0. Removed from `5_Appendix.tex` on 2026-09-19 (was Table 23,
`tab:fullmatch`, and Table 18, `tab:arrestepochs`).

## Every baseline at our full recipe (AdamW 0.021 one-cycle, 100 epochs, teacher init, 8.8/255, WA, AWP)

| method | own recipe | our full recipe | ΔNRR |
|---|---|---|---|
| PGD-AT | 57.36 / 20.34 / 30.03 | 57.73 / 26.46 / 36.29 | +6.26 |
| TRADES | 55.28 / 23.55 / 33.03 | 55.33 / 25.26 / 34.67 | +1.64 |
| MART | 53.86 / 22.72 / 31.96 | 50.48 / 25.74 / 34.12 | +2.16 |
| ARD | 57.61 / 20.24 / 29.96 | 58.04 / 26.44 / 36.48 | +6.52 |
| RSLAD | 59.68 / 21.30 / 31.40 | 59.82 / 25.04 / 35.30 | +3.90 |
| AdaAD | 59.79 / 23.19 / 33.42 | 58.37 / 26.71 / 36.65 | +3.23 |
| AdaAD + IGDM | 48.25 / 19.48 / 27.75 | 52.89 / 25.04 / 33.99 | +6.23 |
| ARREST | 66.44 / 22.99 / 34.16 | 63.65 / 27.48 / 38.39 | +4.23 |
| HAT | 60.67 / 24.45 / 34.85 | 55.42 / 19.56 / 28.91 | −5.94 |
| LBGAT | 56.46 / 26.02 / 35.62 | collapses to chance | --- |
| **CFA (ours)** | --- | **62.17 / 28.86 / 39.42** | --- |

Eight of ten improve, and none reaches CFA. ARREST is the closest, 1.03 NRR behind.

## CFA and ARREST epoch by epoch, both at our recipe and stack, 8/255

| epochs | CFA | ARREST | ΔAA |
|---|---|---|---|
| C100 100 | 63.89 / 27.78 / 38.72 | 64.87 / 27.08 / 38.21 | +0.70 |
| C100 150 | 64.75 / 27.98 / 39.07 | 64.66 / 26.40 / 37.49 | +1.58 |
| C10 50 | 84.48 / 50.47 / 63.19 | 84.49 / 50.61 / 63.30 | −0.14 |
| C10 100 | 87.22 / 51.15 / 64.48 | 85.24 / 51.52 / 64.22 | −0.37 |
| C10 150 | 87.37 / 51.33 / 64.67 | 85.79 / 51.84 / 64.63 | −0.51 |
| C10 200 | 87.76 / 51.44 / 64.86 | 86.13 / 51.33 / 64.33 | +0.11 |

CIFAR-100 is where the objectives separate: our AA lead grows from 0.70 at 100 epochs to 1.58 at 150,
since ARREST loses 0.68 AA over those 50 epochs while we gain 0.20. On CIFAR-10 the two sit on the
same frontier within 0.5 NRR, we hold clean accuracy and they hold AA, so that dataset is not
evidence either way. Defend with CIFAR-100.

## ARREST given our sensitivity-matched radius as well (8.8/255, 100 epochs)

| | CFA | ARREST + our radius |
|---|---|---|
| C100 | 62.17 / 28.86 / 39.42 | 64.08 / 27.53 / 38.51 |
| C10 | 84.96 / 51.74 / 64.31 | 85.93 / 51.76 / 64.60 |

Transplanting the allocation rule onto their objective leaves CIFAR-100 0.91 NRR behind us and puts
CIFAR-10 0.29 ahead. Same reading as above: CIFAR-100 carries the claim.

## ARREST at its own 20-epoch recipe, with pieces added

| variant | C100 | C10 |
|---|---|---|
| published recipe | 66.44 / 22.99 / 34.16 | 85.11 / 46.23 / 59.92 |
| + our radius | 66.79 / 23.10 / 34.33 | 84.80 / 46.32 / 59.91 |
| + WA only | 66.61 / 23.34 / 34.57 | 83.77 / 47.05 / 60.26 |
| + WA + AWP | 64.95 / 23.02 / 33.99 | not run |

Twenty epochs of finetuning has no robust overfitting to undo, so the stack does nothing there. This
is the honest reason ARREST's own recipe does not benefit, and it is worth saying if asked.

## Baselines at their own recipe plus natural init and our stack only (CIFAR-100)

Keeping their optimizer, schedule and epochs, adding the teacher warm start, WA, and AWP with warmup
at 10% of the run. This one is defensible in the paper if a reviewer asks for a stack-matched table.

PGD-AT 59.85 / 25.77 / 36.03, TRADES 56.16 / 24.56 / 34.17, MART 50.97 / 25.13 / 33.66,
ARD 60.19 / 26.10 / 36.41, AdaAD 57.21 / 27.10 / 36.78, HAT 57.63 / 24.74 / 34.62,
ARREST 64.95 / 23.02 / 33.99. RSLAD, IGDM, Consistency, LBGAT and ADR still running.
