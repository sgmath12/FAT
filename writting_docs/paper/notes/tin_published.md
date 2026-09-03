# Tiny-ImageNet-200, ResNet-18, published numbers

2026-09-03. AutoAttack, $\ell_\infty$, $\epsilon = 8/255$, transcribed from the paper and table named
in the last column. Collected the same way as `wrn_published.md`, and with the same warning: read the
source column, because the spread between papers on this dataset is larger than the spread between
methods within any one of them.

| Method | Clean | AA | Avg | NRR | Source |
|---|---:|---:|---:|---:|---|
| **CFA (ours)**, 200-epoch teacher | **55.16** | **20.54** | **37.85** | **29.93** | ours |
| *CFA (ours), 80-epoch teacher* | *57.08* | *18.96* | *38.02* | *28.46* | ours |
| AT + WA + ADR | 48.55 | 20.23 | 34.39 | 28.56 | ADR Tab. 2c |
| AT + WA + AWP + ADR | 48.27 | 20.12 | 34.20 | 28.40 | ADR Tab. 2c |
| TRADES + WA + AWP + ADR | 51.38 | 19.48 | 35.43 | 28.25 | ADR Tab. 4 |
| TRADES + WA + ADR | 51.99 | 19.17 | 35.58 | 28.01 | ADR Tab. 4 |
| TRADES + ADR | 51.82 | 19.17 | 35.50 | 27.99 | ADR Tab. 4 |
| AT + WA + AWP | 48.61 | 19.58 | 34.09 | 27.92 | ADR Tab. 2c |
| AT + ADR | 48.19 | 19.46 | 33.83 | 27.72 | ADR Tab. 2c |
| AT + WA | 49.10 | 19.30 | 34.20 | 27.71 | ADR Tab. 2c |
| Consistency-AT + RPAT | 49.74 | 18.84 | 34.29 | 27.33 | RPAT Tab. 1 |
| **HAT** | **52.60** | 18.14 | 35.37 | 26.98 | ADR Tab. 5 |
| TE | 47.46 | 18.29 | 32.88 | 26.40 | ADR Tab. 5 |
| TRADES + WA | 49.51 | 17.69 | 33.60 | 26.07 | ADR Tab. 4 |
| TRADES + WA + AWP | 49.21 | 17.66 | 33.44 | 25.99 | ADR Tab. 4 |
| AT | 45.87 | 18.06 | 31.96 | 25.92 | ADR Tab. 2c/5 |
| PGD-AT + RPAT | 47.68 | 17.77 | 32.73 | 25.89 | RPAT Tab. 1 |
| TRADES | 48.49 | 17.35 | 32.92 | 25.56 | ADR Tab. 4 |
| Consistency-AT | 46.54 | 17.60 | 32.07 | 25.54 | RPAT Tab. 1 |
| TRADES + RPAT | 48.77 | 16.92 | 32.84 | 25.12 | RPAT Tab. 1 |
| MART + RPAT | 41.76 | 17.79 | 29.77 | 24.95 | RPAT Tab. 1 |
| PGD-AT | 46.32 | 17.07 | 31.70 | 24.95 | RPAT Tab. 1 |
| TRADES | 46.75 | 16.60 | 31.68 | 24.50 | RPAT Tab. 1 |
| MART | 39.70 | 17.18 | 28.44 | 23.98 | RPAT Tab. 1 |
| *F$^2$AT* | *40.54* | *13.13* | *26.84* | *19.84* | F$^2$AT Tab. V |
| *SEAT* | *35.51* | *11.37* | *23.44* | *17.22* | F$^2$AT Tab. V |
| *MART* | *33.57* | *10.39* | *21.98* | *15.87* | F$^2$AT Tab. V |
| *SAT* | *31.52* | *9.37* | *20.45* | *14.45* | F$^2$AT Tab. V |
| *TRADES* | *36.19* | *8.26* | *22.22* | *13.45* | F$^2$AT Tab. V |

## Our row beats every one of them on both axes

At $55.16 / 20.54$ the shipped recipe is strictly better on clean accuracy **and** AutoAttack than all
$27$ published rows. Nothing published is above it on either axis taken alone: the highest clean is
HAT's $52.60$ and the highest AutoAttack is $20.23$ from AT + WA + ADR. NRR $29.93$ against the best
published $28.56$.

This is the only one of our three datasets where that is true. On CIFAR-10 WideResNet we dominate
$30$ of $31$ rows with ARREST outside; on CIFAR-100 we have not run the WideResNet cell at all.

The 80-epoch-teacher variant is reported as the teacher-length ablation, not the headline: it trades
$1.58$ AutoAttack for $1.92$ clean and then loses the AutoAttack axis to eight ADR rows, so it
dominates only $19$ of $27$.

## Caveats, and one big one

- **F$^2$AT's block is not comparable to the rest and is italicised for that reason.** Its plain SAT
  baseline reaches $31.52 / 9.37$ where ADR's plain AT reaches $45.87 / 18.06$ — a $14$-point gap in
  clean and $9$ in AutoAttack for what is nominally the same baseline on the same architecture and
  dataset. Something differs in the setup that neither paper states; do not mix the two blocks.
- **ADR trains Tiny-ImageNet for 80 epochs, not 200**, and crops to $64 \times 64$. We train $100$.
  That is in our favour by $20$ epochs and should be stated where the row is used.
- ADR's Table 5 credits its HAT and TE rows to Rade & Moosavi-Dezfooli (2022) and Dong et al. (2022a);
  they are ADR's transcription, not our measurement, and we have HAT ported so this row can eventually
  become ours.
- **LBGAT reports Tiny-ImageNet but without AutoAttack** — its Table 6 is PGD-20 only, at clean
  $30.65$–$38.51$, far below every row above. Excluded rather than mixed in.
- **B-MTARD's Tiny-ImageNet is MobileNet-v2**, not ResNet-18, so it does not belong here.
- CURE, ARREST and Generalist++ report no Tiny-ImageNet results at all.
