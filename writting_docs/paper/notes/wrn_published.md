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

**On CIFAR-10 one method already leads both metrics.** AT + WA + AWP tops Avg at $71.22$ and NRR at
$67.53$, with RPAT++ and ADR + WA + AWP within $0.6$ and $0.3$ behind. The whole field sits between
$52$ and $55$ AutoAttack at $85$–$88$ clean; the top three Avg values span $0.53$. Winning there means
moving a decimal against a settled ranking.

**On CIFAR-100 the two metrics disagree, and nobody leads both:**

| | Avg | NRR |
|---|---|---|
| 1st | ARREST $48.69$ | ADR + WA + AWP $41.91$ |
| 2nd | LBGAT $48.54$ | AT + WA + AWP $41.33$ |
| ARREST's rank on the other | — | **11th of 13** ($36.49$) |
| ADR's rank on the other | 4th ($46.91$) | — |

The field is bimodal. One end raises clean and gives up robustness (ARREST $73.05/24.32$, LBGAT
$70.03/27.05$); the other raises robustness and gives up clean (ADR + WA + AWP $62.21/31.60$). Between
clean $63.40$ and clean $70.03$ there is **nothing at all**, and the two leaders are at opposite ends
of that hole.

**The opening is a cell that leads both metrics at once, which no published result does.** It requires
clearing Avg $48.69$ and NRR $41.91$ together, which the two constraints put at roughly:

| clean | AA needed |
|---|---|
| 66 | $\geq 31.4$ |
| 67 | $\geq 30.5$ |
| 68 | $\geq 30.3$ |
| 70 | $\geq 30.0$ |

⚠ Note what this corrects: a cell at $68/30$ does **not** dominate ARREST or LBGAT on both axes --
$68 < 70.03 < 73.05$ -- and at NRR $41.63$ it does not even lead ADR. Domination of the clean end is
not available at these numbers; leading both *metrics* is.

**Is it reachable?** ADR's own ResNet-18 $\to$ WideResNet gain on this dataset is $+4.85$ clean and
$+3.10$ AA ($57.36/28.50 \to 62.21/31.60$). Applying the same delta to our ResNet-18 result of
$62.17/28.86$ projects $67.02/31.96$, which is Avg $49.49$ and NRR $43.28$ -- clearing both. That is a
projection from one method's scaling and not a result, but it is the reason to spend WideResNet
compute on CIFAR-100 before CIFAR-10.

It is also where our claim is easiest to state, since ARREST and LBGAT are both natural-teacher
methods and both sit on the clean end of exactly this hole.

## Our CIFAR-10 result, measured 2026-09-03

`wrn_champ_freezehead`, CIFAR-10, run on the other server at the shipped recipe. Reported order is
clean, FGSM, PGD-10/20/50, CW, AutoAttack.

| | clean | FGSM | PGD-10 | PGD-20 | PGD-50 | CW | **AA** | Avg | **NRR** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **CFA, WRN-34-10** | **88.67** | 63.01 | 57.05 | 56.08 | 55.83 | 56.79 | **55.29** | **71.98** | **68.11** |

PGD is monotone in attack strength as it should be, $57.05 \ge 56.08 \ge 55.83$.

### Against the WideResNet baselines

Ranked by NRR against every published WRN-34-10 row collected above:

| | Method | clean | AA | Avg | NRR | Source |
|---:|---|---:|---:|---:|---:|---|
| **1** | **CFA (ours)** | **88.67** | **55.29** | **71.98** | **68.11** | — |
| 2 | AT + WA + AWP | 87.42 | 55.01 | 71.22 | 67.53 | ADR |
| 3 | ADR + WA + AWP | 86.11 | 55.26 | 70.69 | 67.32 | ADR |
| 4 | RPAT++ | 86.76 | 54.97 | 70.87 | 67.30 | RPAT |
| 5 | ReBAT | 85.25 | 54.78 | 70.02 | 66.70 | RPAT |
| 6 | KD + SWA | 87.45 | 53.59 | 70.52 | 66.46 | RPAT |
| 7 | S2O | 85.67 | 54.10 | 69.89 | 66.32 | ARREST |
| 8 | AWP | 85.57 | 54.04 | 69.80 | 66.24 | ARREST |
| 9 | LBGAT ($\alpha{=}0$) | 88.22 | 52.86 | 70.54 | 66.11 | LBGAT |
| 10 | LAS-AT | 86.23 | 53.58 | 69.91 | 66.09 | ARREST |

Against the 31 published rows:

- **Highest AutoAttack**, $55.29$ against the previous best $55.26$ (ADR + WA + AWP). Nothing published
  is higher.
- **Highest Avg** ($71.98$ against $71.22$) and **highest NRR** ($68.11$ against $67.53$), both
  previously held by AT + WA + AWP.
- **Strictly better on both axes than 30 of the 31 rows.** The one exception is ARREST, at $1.57$ more
  clean and $5.09$ less AutoAttack — a different point on the trade-off, not a domination either way.
- Second-highest clean accuracy in the table, behind only ARREST.

The two natural-teacher baselines are the informative comparison, since they are the methods closest
to ours in what they assume available. We are $+0.45$ clean and $+2.43$ AutoAttack over LBGAT
($88.22/52.86$), and $-1.57$ clean and $+5.09$ AutoAttack over ARREST ($90.24/50.20$). Both of them
buy their clean accuracy with robustness; this does not.

## The architecture is the same in all five papers, and in ours

Checked 2026-09-03 rather than assumed. ADR, CURE, RPAT and ARREST all state **WRN-34-10**
explicitly. LBGAT's main models are WRN-34-10 too ("the WRN-34-10 architecture [54] are adopted",
its Sec. 4); the starred rows in its Table 5 are WRN-34-20 and are excluded from the tables above.

Ours is the same network despite reporting a different parameter count. `scripts/check_arch.py`
prints $48.3$M where the literature says $46.2$M, and the whole difference is a `sub_block1` in our
`WideResNet.py`: a duplicate of `block1` that is constructed but never referenced in `forward`, and
measured to receive exactly zero gradient. It is $2.10$M parameters, and $48.26 - 2.10 = 46.16$M,
which matches RPAT's and HAT's implementations to the second decimal. The dead block comes from the
TRADES codebase and LBGAT's released WideResNet has it as well, at the same $48.26$M. **The network
we train is WRN-34-10 and our numbers are comparable with every row above.** Do not delete
`sub_block1` to tidy the count -- existing WideResNet checkpoints carry those keys.

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
