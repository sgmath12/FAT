# The table grid, and what is missing from it

2026-09-02. Five result tables, one method list, so that a cell is either filled or visibly empty.
`\Cref{tab:main}` carries CIFAR-10 and CIFAR-100 on ResNet-18; `tab:tin`, `tab:w10` and `tab:w100`
carry the other three slots and are laid out with the same eleven rows.

## Current state

| | C10/RN18 | C100/RN18 | TIN/RN18 | C10/WRN | C100/WRN |
|---|---|---|---|---|---|
| PGD-AT | queued | queued | — | — | — |
| TRADES | queued | queued | — | — | — |
| MART | queued | queued | — | — | — |
| PGD-AT @ teacher init | ✅ | ✅ | — | — | — |
| ARD | ✅ | ✅ | — | — | — |
| RSLAD | ✅ | ✅ | — | — | — |
| AdaAD | ✅ | ✅ | — | — | — |
| AdaAD + IGDM | ✅ | ✅ | — | — | — |
| ADR | — | — | — | — | — |
| CURE | ✗ does not reproduce | ✗ | ✗ | ✗ | ✗ |
| RPAT++ | ✅ | ✅ | — | — | — |
| **CFA (ours)** | ✅ | ✅ | ✅ *(other server)* | running *(other server)* | running *(other server)* |

**33 empty cells, ≈329 GPU-hours** if every one is filled. Per-cell cost relative to CIFAR/ResNet-18:
Tiny-ImageNet ×4.0 (64×64, 100k images), WideResNet-34-10 ×4.5.

## ⚠ The Tiny-ImageNet result is not on this machine

`featdir_tin_100ep` (55.16 / 20.54, 200-epoch teacher) is documented but its log lives on the other
server, so `results/TinyImageNet/` here has no AutoAttack line for it. **Copy that log across**, or the
number cannot be regenerated from this repository.

## If the whole grid is not affordable, this is the order

Four rows carry the argument; the other seven are completeness.

| row | why it is needed on every dataset |
|---|---|
| **PGD-AT** | the reference point the trade-off is measured against |
| **AdaAD** | the strongest distillation objective on a natural teacher — "even this one falls below not distilling" |
| **PGD-AT @ teacher init** | the "do not distil at all" upper bound, which is what all four fall below |
| **CFA (ours)** | ours |

That is 12 cells rather than 33, ≈104 GPU-hours, and it preserves every claim the tables are used for.
The remaining rows (TRADES, MART, ARD, RSLAD, IGDM, ADR, RPAT++) are already present on both
ResNet-18/CIFAR slots, which is where the paper's arguments are made.

## Also queued, and not part of the grid

`champ_eps6` and `champ_eps7` on both CIFAR datasets, which extend the operating curve of Figure 1
rather than filling a table. At $\varepsilon_{\mathrm{tr}} = 8/255$ the curve already dominates
Generalist++ on CIFAR-100 (63.89 / 27.78 against 62.97 / 23.96, both axes) and CURE and ARREST on
CIFAR-10; what it does not yet reach is Generalist++'s CIFAR-10 clean accuracy of 89.09, and the
measured trend — clean falls ≈1.9 points per +1/255 while AutoAttack is flat — puts $6/255$ near
clean 91 at an AA far above 46.07.
