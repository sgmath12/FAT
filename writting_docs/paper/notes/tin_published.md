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
| TE\* | 47.46 | 18.29 | 32.88 | 26.40 | ADR Tab. 5 |
| TRADES + WA | 49.51 | 17.69 | 33.60 | 26.07 | ADR Tab. 4 |
| TRADES + WA + AWP | 49.21 | 17.66 | 33.44 | 25.99 | ADR Tab. 4 |
| AT | 45.87 | 18.06 | 31.96 | 25.92 | ADR Tab. 2c/5 |
| TRADES | 48.49 | 17.35 | 32.92 | 25.56 | ADR Tab. 4 |
| *(below: PreActResNet-18, 11.27M --- same size, different network)* | | | | | |
| *HAT* | *52.60* | *18.14* | *35.37* | *26.98* | HAT Tab. 9 |
| *TRADES* | *48.25* | *17.17* | *32.71* | *25.35* | HAT Tab. 9 |
| *AT* | *47.76* | *17.92* | *32.84* | *26.02* | HAT Tab. 9 |
| *(below: Bottleneck ResNet-18, 14.35M --- see the architecture section)* | | | | | |
| *Consistency-AT + RPAT* | *49.74* | *18.84* | *34.29* | *27.33* | RPAT, 14.35M |
| *PGD-AT + RPAT* | *47.68* | *17.77* | *32.73* | *25.89* | RPAT, 14.35M |
| *Consistency-AT* | *46.54* | *17.60* | *32.07* | *25.54* | RPAT, 14.35M |
| *TRADES + RPAT* | *48.77* | *16.92* | *32.84* | *25.12* | RPAT, 14.35M |
| *MART + RPAT* | *41.76* | *17.79* | *29.77* | *24.95* | RPAT, 14.35M |
| *PGD-AT* | *46.32* | *17.07* | *31.70* | *24.95* | RPAT, 14.35M |
| *TRADES* | *46.75* | *16.60* | *31.68* | *24.50* | RPAT, 14.35M |
| *MART* | *39.70* | *17.18* | *28.44* | *23.98* | RPAT, 14.35M |
| *F$^2$AT* | *40.54* | *13.13* | *26.84* | *19.84* | F$^2$AT Tab. V |
| *SEAT* | *35.51* | *11.37* | *23.44* | *17.22* | F$^2$AT Tab. V |
| *MART* | *33.57* | *10.39* | *21.98* | *15.87* | F$^2$AT Tab. V |
| *SAT* | *31.52* | *9.37* | *20.45* | *14.45* | F$^2$AT Tab. V |
| *TRADES* | *36.19* | *8.26* | *22.22* | *13.45* | F$^2$AT Tab. V |

## Our row beats every one of them on both axes

At $55.16 / 20.54$ the shipped recipe is strictly better on clean accuracy **and** AutoAttack than all
$27$ published rows. Nothing published is above it on either axis taken alone: among the rows that are the same network as ours the highest clean is TRADES + WA + ADR's
$51.99$ and the highest AutoAttack is $20.23$ from AT + WA + ADR. NRR $29.93$ against the best
published $28.56$.

This is the only one of our three datasets where that is true. On CIFAR-10 WideResNet we dominate
$30$ of $31$ rows with ARREST outside; on CIFAR-100 we have not run the WideResNet cell at all.

The 80-epoch-teacher variant is reported as the teacher-length ablation, not the headline: it trades
$1.58$ AutoAttack for $1.92$ clean and then loses the AutoAttack axis to eight ADR rows, so it
dominates only $19$ of $27$.

## Which rows are ours

**Exactly one: the CFA row.** Everything else is transcribed from the paper in the source column. We
have run no Tiny-ImageNet baseline ourselves — the porting work (ADR, HAT, LBGAT, Consistency-AT) is
done and the configs exist, but Tiny-ImageNet needs the 200-epoch teacher and is queued for the other
machine.

## The architecture is NOT the same in all of them

Checked in each repository rather than taken from the word "ResNet-18" in a caption, and it does not
survive the check.

| Source | what its "ResNet-18" is | params (200 classes) |
|---|---|---|
| **ours** | `BasicBlock`, standard ResNet-18 | **11.27 M** |
| **ADR** | `resnet18` in `create_model.py`, standard | 11.27 M |
| **RPAT** | `resnet18 = ResNet(Bottleneck, [2,2,2,2])` | **14.35 M** |
| *(RPAT also ships)* | `pre_resnet18 = ResNet(PreActBlock, ...)` | 11.27 M |

**HAT's Tiny-ImageNet row is PreActResNet-18.** ADR's Table 5 lists
"Rade \& Moosavi-Dezfooli (2022) $18.14 / 52.60$" under a heading that says ResNet-18, but those are
HAT's own numbers quoted verbatim from its Table 9, whose caption reads "Comparison of HAT using
\textbf{PreAct ResNet-18} on TinyImagenet-200". HAT's released code makes the point for us: on this
dataset it asserts `'preact-resnet' in name` and refuses anything else. PreActResNet-18 has the same
$11.27$M parameters as ours, so it is far closer than RPAT's, but it is a different network and it is
labelled as one. HAT's Table 9 also gives AT $47.76/17.92$ and TRADES $48.25/17.17$ on the same
network, which are added rather than dropped since they make that block self-contained.

⚠ **TE's row is unverified.** ADR credits $47.46 / 18.29$ to Dong et al. (2022a) and we do not have
that paper, so whether it is a standard ResNet-18 cannot be checked the way the others were. It is
marked with an asterisk and should not carry weight.

**RPAT's Table 1 is a $14.35$M Bottleneck network, $28\%$ larger than ours.** Its paper says
"ResNet-18 [15]", citing He et al., and its code's `resnet18` uses `Bottleneck` where a standard
ResNet-18 uses `BasicBlock`. They ship a `pre_resnet18` at $11.27$M too and use PreActResNet-18
explicitly for their Tables 3 and 4, so the distinction is theirs and deliberate; Table 1 is the
Bottleneck one. Its rows are therefore separated below rather than mixed in.

ADR checks out: its Tiny-ImageNet gin configs all set `create_model.model_name = "resnet18"`, and
`create_model.py` routes that to the standard `ResNet` while routing `"preact-resnet18"` elsewhere,
so those rows are the same network as ours at the same parameter count.

For the WideResNet table the equivalent check passed — ARREST states "We used WideResNet-34-10 as the
main DNN architecture", and the other four say WRN-34-10 explicitly.

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
