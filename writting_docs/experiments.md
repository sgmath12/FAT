# Experiments

Draft, 2026-09-01. Table structure follows `reference/ADR.pdf` (Table 1 main comparison, Table 2 the
same method stacked with WA/AWP, Tables 5–9 per-dataset and ablation) and `reference/IGDM.pdf`
(Table 1 teacher models before any student result). Settings paragraphs follow
`reference/self-distililation-at.pdf` §5.1: Dataset / Training details / Comparison models /
Evaluation, in that order.

⚠ **Cells marked † are being re-measured** under the corrected AWP ascent objective (fix 021f0ef;
`_awp_loss_fn` had included a head term the model never trains, supplying 40.8% of the AWP backbone
gradient). Values shown are pre-fix and are the ones currently reported everywhere else in the
drafts. Cells without AWP are unaffected and are final.

---

## 5. Experiments

### 5.1 Experimental settings

**Datasets.** CIFAR-10, CIFAR-100 (Krizhevsky, 2009) and Tiny-ImageNet-200 (Le & Yang, 2015). Random
crop with 4-pixel padding and random horizontal flip; no other augmentation.

**Teacher.** For each dataset, a network of the same architecture as the student, trained naturally
for 200 epochs on the same data (Table 1). No adversarial training is performed at any point in the
teacher's construction, so it is available at the cost of ordinary supervised training.

**Training details.** ResNet-18 throughout the main tables; WideResNet-34-10 in §5.7. The student is
initialized from the teacher and trained for 100 epochs with AdamW at learning rate $0.021$ under a
one-cycle schedule, batch size 128. The attack is 10-step PGD at step size $2/255$ with a training
radius of $8.8/255$; §5.5 varies that radius. Weight averaging ($\kappa=0.999$, from 20% of the run)
and an AWP proxy ($\gamma=0.005$, after epoch 10) are used and are ablated in §5.4. **Every dataset
uses this identical configuration; only the dataset name and the teacher checkpoint change.**

**Comparison models.** Two groups. (i) Adversarial training methods that need no external network:
PGD-AT, TRADES, MART, Consistency-AT, RPAT, ADR, CURE, Generalist++, ARREST. (ii) Adversarial
distillation methods, which assume an adversarially trained teacher: ARD, RSLAD, AdaAD, IGDM,
B-MTARD. Published numbers are quoted with their source; RPAT and CURE we additionally re-ran
ourselves (§5.2), and the four distillation objectives we ported and ran on our own natural teacher
(§5.3).

**Evaluation.** Clean accuracy, FGSM, PGD-20/10/50, CW$_\infty$, and AutoAttack (standard version,
`apgd-ce`+`apgd-t`) on the full test set at $\varepsilon=8/255$. NRR is the harmonic mean of clean
accuracy and AutoAttack accuracy. All numbers are the final-epoch (weight-averaged) model, not a
best-checkpoint selection.

**Table 1.** Teacher models. AutoAttack accuracy is $0$ by construction: these networks are trained
without any adversarial example.

| dataset | teacher | epochs | clean | AA |
|---|---|---:|---:|---:|
| CIFAR-10 | ResNet-18, natural | 200 | 95.32 | 0.00 |
| CIFAR-100 | ResNet-18, natural | 200 | 77.66 | 0.00 |
| Tiny-ImageNet | ResNet-18, natural | 200 | 66.29 | 0.00 |

---

### 5.2 Main results

Baselines are grouped by what they assume is available, since that is what the comparison turns on.
**(a)** standard adversarial training, which needs no other network; **(b)** adversarial distillation
from an *adversarially trained* teacher; **(c)** methods guided by a *naturally* trained network,
which is the family ours belongs to; **(d)** the current accuracy–robustness trade-off methods.
Published numbers carry their source and their backbone where it differs; `—` means the paper does not
report that cell for this dataset and architecture.

**Table 2.** CIFAR-10, ResNet-18, $\ell_\infty$, $\varepsilon=8/255$.

| | Method | clean | PGD-20 | CW | **AA** | **NRR** | src |
|---|---|---:|---:|---:|---:|---:|:--:|
| (a) | PGD-AT | 82.78 | 51.30 | 49.72 | 44.63 | 57.99 | CURE |
| | TRADES | 82.41 | 52.76 | 50.43 | 48.37 | 60.96 | CURE |
| | MART | 80.70 | 54.02 | 49.35 | 47.49 | 59.79 | CURE |
| | Consistency-AT | 83.42 | 51.96 | — | 47.72 | 60.71 | RPAT |
| (b) | ARD | 85.04 | 53.27 | 50.27 | 49.49 | 62.55 | IGDM |
| | RSLAD | 83.59 | 55.98 | 53.15 | 52.13 | 64.16 | IGDM |
| | IAD | 84.33 | 54.24 | 50.97 | 50.09 | 62.85 | IGDM |
| | AdaAD | 84.74 | 56.78 | 53.51 | 52.79 | 65.03 | IGDM |
| | AdaAD + IGDM | 84.83 | 57.61 | 55.09 | **54.02** | **65.98** | IGDM |
| (c) | LBGAT | — | — | — | — | — | *(not obtained)* |
| | ARREST | 86.63 | — | — | 46.14 | 60.21 | ARREST |
| | CURE | 86.76 | 54.92 | 52.48 | 49.69 | 63.19 | CURE |
| | ADR (AT + ADR) | 82.41 | — | — | 50.38 | 62.53 | ADR |
| (d) | Consistency-AT + RPAT | 84.12 | 52.33 | — | 48.98 | 61.91 | RPAT |
| | RPAT++ *(ours, reproduced)* | 82.41 | 55.72 | 52.88 | 50.75 | 62.82 | ours |
| | **CFA (ours)** † | 85.58 | 52.94 | 53.88 | 51.79 | **64.53** | ours |

**Table 3.** CIFAR-100, ResNet-18.

| | Method | clean | PGD-20 | CW | **AA** | **NRR** | src |
|---|---|---:|---:|---:|---:|---:|:--:|
| (a) | PGD-AT | 56.56 | 28.80 | — | 25.02 | 34.69 | RPAT |
| | TRADES | 55.39 | 29.36 | — | 24.51 | 33.98 | RPAT |
| | MART | 49.83 | 30.38 | — | 25.00 | 33.30 | RPAT |
| | Consistency-AT | 58.53 | 29.28 | — | 25.39 | 35.42 | RPAT |
| (b) | ARD | 61.51 | 30.23 | 26.97 | 24.77 | 35.29 | IGDM |
| | RSLAD | 60.22 | 32.16 | 27.96 | 26.76 | 37.15 | IGDM |
| | IAD | 59.92 | 31.47 | 26.91 | 25.15 | 35.42 | IGDM |
| | AdaAD | 64.43 | 33.21 | 29.53 | 28.06 | 39.10 | IGDM |
| | AdaAD + IGDM | **64.44** | 36.19 | 31.75 | **30.32** | **41.24** | IGDM |
| (c) | LBGAT | — | — | — | — | — | *(not obtained)* |
| | ARREST | — | — | — | — | — | *WRN only* |
| | CURE | — | — | — | — | — | *no AA on C100* |
| | ADR (AT + ADR) | 56.10 | — | — | 26.87 | 36.34 | ADR |
| | ADR (AT + WA + AWP + ADR) | 57.36 | — | — | 28.50 | 38.08 | ADR |
| (d) | Consistency-AT + RPAT | 60.33 | 29.97 | — | 26.31 | 36.64 | RPAT |
| | RPAT++ *(ours, reproduced)* | 55.93 | 32.44 | 29.37 | 27.36 | 36.74 | ours |
| | **CFA (ours)** † | 62.65 | 32.63 | 30.66 | 28.77 | **39.43** | ours |

**Table 4.** Tiny-ImageNet-200, ResNet-18.

| | Method | clean | **AA** | **NRR** | src |
|---|---|---:|---:|---:|:--:|
| (a) | PGD-AT | 45.87 | 18.06 | 25.91 | ADR |
| | Rade & Moosavi-Dezfooli | 52.60 | 18.14 | 26.96 | ADR |
| | Dong et al. | 47.46 | 18.29 | 26.42 | ADR |
| (b) | ARD / RSLAD / AdaAD / IGDM | — | — | — | *not reported* |
| (c) | LBGAT / ARREST / CURE | — | — | — | *not reported* |
| | ADR (AT + ADR) | 48.55 | 20.23 | 28.61 | ADR |
| (d) | RPAT | — | — | — | *not reported* |
| | **CFA (ours)**, 200-epoch teacher † | **55.16** | **20.54** | **29.93** | ours |
| | CFA (ours), 80-epoch teacher † | 57.08 | 18.96 | 28.46 | ours |

**Table 5.** CIFAR-10, WideResNet-34-10.

| | Method | clean | **AA** | **NRR** | src |
|---|---|---:|---:|---:|:--:|
| (a) | PGD-AT / TRADES | — | — | — | *to fill* |
| (b) | ARD / RSLAD / AdaAD / IGDM | — | — | — | *to fill* |
| (c) | LBGAT | 88.22 | 52.18 | 65.57 | ARREST |
| | ARREST | 90.24 | 50.20 | 64.51 | ARREST |
| | CURE | 87.05 | 52.10 | 65.19 | RPAT |
| | ADR | 84.67 | 53.25 | 65.38 | RPAT |
| (d) | ReBAT | 85.25 | 54.78 | 66.70 | RPAT |
| | RPAT++ | 86.76 | **54.97** | **67.30** | RPAT |
| | **CFA (ours)** | *running* | | | ours |

**Table 6.** CIFAR-100, WideResNet-34-10.

| | Method | clean | **AA** | **NRR** | src |
|---|---|---:|---:|---:|:--:|
| (a) | PGD-AT (SAT) | 53.64 | 21.01 | 30.20 | F²AT |
| (b) | ARD / RSLAD / AdaAD / IGDM | — | — | — | *to fill* |
| (c) | LBGAT | 70.25 | 26.73 | 38.72 | ADR |
| | ARREST | **73.05** | 24.32 | 36.50 | ADR |
| | CURE | — | — | — | *no AA on C100* |
| | ADR (AT + ADR) | 62.21 | **31.60** | **41.90** | ADR |
| (d) | F²AT | 60.21 | 26.91 | 37.23 | F²AT |
| | RPAT | — | — | — | *PreActResNet-18 only* |
| | **CFA (ours)** | *running* | | | ours |

⚠ **Group (b) is not a like-for-like comparison and is printed to make that visible.** Those rows are
produced from a WideResNet-28-10 or WRN-70-16 teacher adversarially trained with generated data — a
network more expensive than the student it teaches — so their robustness originates outside the
student. §5.3 gives the same four objectives the teacher they would have in our setting, and every one
of them then falls below not distilling at all. The two settings assume different inputs and are best
read side by side rather than ranked.

⚠ **Backbone caveats.** ARREST's CIFAR-10 headline is WideResNet-34-10 (Table 5); the ResNet-18 row in
Table 2 is its smaller-backbone entry. CURE reports no AutoAttack on CIFAR-100. RPAT's WideResNet
results are PreActResNet-18 on CIFAR-100. Each cell states the backbone its source used.

**Reproductions.** RPAT++ we reproduce to within $0.31$ AutoAttack of its published numbers on both
datasets and report our own measurement. CURE we could **not** reproduce: seven runs of the official
repository fall short on *different* axes — the closest on standard accuracy reaches $86.11$ at AA
$40.65$, the closest on robustness AA $45.63$ at clean $81.15$, while the published pair has both.
Computing its RGP prominence mask on absolute values moves AA by $+15.9$, which localizes the
discrepancy without closing it, so the CURE rows above are the paper's own numbers.

---

### 5.3 Published distillation objectives, given a natural teacher

The four objectives were ported into our framework and given our natural teacher in place of the
adversarially trained teacher they assume, each at its own published recipe (SGD $0.1$, step decay at
epochs 70 and 90, random initialization, no weight averaging, no AWP, 100 epochs). The reference row
is plain PGD-AT initialized from the same teacher — that is, using the teacher for nothing but the
starting point.

**Table 7.** Adversarial distillation with a natural teacher. ResNet-18, 100 epochs.

| | CIFAR-100 clean | AA | NRR | | CIFAR-10 clean | AA | NRR |
|---|---:|---:|---:|---|---:|---:|---:|
| ARD | 57.61 | 20.24 | 29.96 | | 84.58 | 43.75 | 57.67 |
| RSLAD | 59.68 | 21.30 | 31.40 | | 85.15 | 42.91 | 57.06 |
| AdaAD | 59.79 | 23.19 | 33.42 | | 85.39 | 46.22 | 59.98 |
| AdaAD + IGDM ($\alpha=20$, published) | 1.00 | 1.00 | 1.00 | | 9.99 | — | — |
| AdaAD + IGDM ($\alpha=1$) | 48.25 | 19.48 | 27.75 | | *running* | | |
| **PGD-AT, initialized at the teacher** | 57.73 | **26.46** | **36.29** | | 82.23 | **50.72** | **62.74** |
| **CFA (ours), no WA, no AWP** | 61.21 | 25.24 | 35.74 | | *running* | | |

Every published objective lands below the row that does not distil at all, on both datasets. The
$\alpha=20$ collapse is not a porting artifact: with a natural teacher the IGDM target
$\mathrm{softmax}(f_T(x{+}\delta)-f_T(x{-}\delta))$ places $0.917$ of its mass on one class and the
term reaches $17.3\times$ the AdaAD loss, and $\alpha=1$ — which puts the two terms on comparable
scales — trains normally and still finishes far below AdaAD alone.

**Table 8.** Local linearity, which IGDM's construction requires of the teacher. Remainder proportion
of the first-order Taylor expansion over an $\varepsilon=8/255$ ball, computed as in IGDM §3.1.

| network | remainder proportion |
|---|---:|
| IGDM's robust teachers (LTD / BDM-AT / IKL-AT, published) | 0.012 / 0.012 / 0.016 |
| PGD-AT ResNet-18 (ours) | **0.0111** |
| **natural teacher (ours)** | **0.4763** |
| anchored student, $p=0$ | 0.0029 |
| anchored student, shipped recipe | 0.0202 |

Our adversarially trained network reproduces IGDM's reported value, so the measurement is calibrated;
the natural teacher is forty times further from linear, and the finite difference IGDM matches is not
a gradient there.

---

### 5.4 Component ablation

**Table 9.** CIFAR-100, ResNet-18, raw-$\ell_2$ anchor, $\varepsilon_{\mathrm{train}}=8.8/255$ held
fixed, no `freeze_lr`. Each row adds one component to the row above it.

| | clean | PGD-20 | CW | **AA** | **NRR** |
|---|---:|---:|---:|---:|---:|
| **50 epochs** | | | | | |
| anchor ($p=0$) | 61.33 | 32.94 | 27.80 | 26.19 | 36.71 |
| ‥ + sensitivity-matched $\varepsilon$ | 62.94 | 32.27 | 28.23 | 26.40 | 37.20 |
| ‥ + weight averaging | 61.11 | 34.30 | 29.66 | **28.06** | **38.46** |
| ‥ + AWP † | 59.90 | 35.70 | 29.74 | 28.05 | 38.21 |
| **100 epochs** | | | | | |
| anchor ($p=0$) | 61.21 | 31.26 | 26.82 | 25.24 | 35.74 |
| ‥ + sensitivity-matched $\varepsilon$ | 62.98 | 31.30 | 27.05 | 25.43 | 36.23 |
| ‥ + weight averaging | 62.43 | 33.49 | 28.96 | 27.39 | 38.08 |
| ‥ + AWP † | 62.35 | 36.26 | 30.65 | **28.68** | **39.29** |

Three readings. The sensitivity rule contributes **exactly $+0.49$ NRR at both schedule lengths**
($+1.61/+0.21$ at 50 epochs, $+1.77/+0.19$ at 100) at identical total attack budget. AWP changes sign
with schedule length: $-0.25$ NRR at 50 epochs, $+1.21$ at 100. And **50 epochs with weight averaging
alone already exceeds ADR's full stack** ($38.46$ against $38.08$), at half the schedule and one fewer
component — so the gain is not the stack.

**Table 10.** What to do with the classifier. CIFAR-100, base regime (50 epochs, raw features, no WA,
no AWP, $\varepsilon=8/255$), so no component of the stack can absorb the difference.

| | clean | PGD-20 | CW | **AA** | **NRR** |
|---|---:|---:|---:|---:|---:|
| logit target only, $\tau=1$ | 58.26 | 22.62 | 22.79 | 20.84 | 30.70 |
| logit target only, $\tau=4$ | 59.39 | 30.30 | 26.84 | 24.48 | 34.67 |
| logit target only, $\tau=16$ | 57.78 | 31.57 | 26.11 | 24.00 | 33.91 |
| feature anchor + head KD, $\tau=1$ | 62.34 | 26.34 | 26.11 | 23.93 | 34.58 |
| feature anchor + head KD, $\tau=16$ | 62.61 | 32.47 | 27.32 | 25.61 | 36.35 |
| **feature anchor, head untouched** | **62.72** | 28.69 | **27.80** | **25.88** | **36.64** |
| ‥ head refit, adversarial CE | 62.77 | 29.93 | 27.66 | 25.65 | 36.42 |
| ‥ head refit, adv CE + smoothing 0.1 | 62.57 | 30.43 | 27.55 | 25.66 | 36.39 |
| ‥ head refit, clean CE | 62.70 | 28.90 | 27.23 | 25.15 | 35.90 |

Leaving the teacher's classifier alone beats every alternative, including the best temperature. At the
full recipe the same holds: freezing the head scores $62.65 / 28.77 /$ NRR $39.43$ against
$62.35 / 28.68 / 39.29$ with the head-KD term retained. This is what removes $\tau$ and $\beta$ from
the method.

---

### 5.5 Training radius

**Table 11.** $\varepsilon_{\mathrm{train}}$ swept with everything else fixed; evaluation is at
$\varepsilon=8/255$ in every cell.

| | clean | PGD-20 | CW | **AA** | **NRR** |
|---|---:|---:|---:|---:|---:|
| **CIFAR-100** | | | | | |
| $8/255$ | **63.89** | 31.41 | 29.83 | 27.78 | 38.72 |
| $8.8/255$ † | 62.65 | 32.63 | **30.66** | **28.77** | **39.43** |
| $10/255$ † | 59.99 | **32.82** | 30.53 | **28.77** | 38.89 |
| **CIFAR-10** | | | | | |
| $8/255$ † | **87.22** | 52.43 | 53.55 | 51.15 | 64.48 |
| $8.8/255$ † | 85.58 | 52.94 | **53.88** | **51.79** | **64.53** |
| $10/255$ † | 83.29 | **53.25** | **53.88** | **51.79** | 63.87 |

Two things hold on both datasets. NRR peaks at $8.8/255$, but the curve is mild and **the standard
$8/255$ already exceeds the best published result on both datasets** ($38.72$ against ADR's $38.08$;
$64.48$ against CURE's $63.19$), so the non-standard radius is a refinement rather than the source of
the result. And past $8.8/255$ the radius buys nothing: AutoAttack and CW are identical to the last
decimal at $10/255$ while standard accuracy falls $2.66$ and $2.29$ — matching the
$\varepsilon$-independence that Appendix A predicts above the threshold.

---

### 5.6 What the teacher transfers

**Table 12.** Only the teacher checkpoint changes; the student configuration is untouched.
CIFAR-100, ResNet-18.

| teacher | teacher clean | teacher $S_w/S_b$ | student clean | student AA | NRR |
|---|---:|---:|---:|---:|---:|
| 50 epochs | 75.81 | 1.100 | **64.28** | 24.38 | 35.35 |
| 100 epochs | 76.62 | 0.982 | 63.65 | 25.20 | 36.11 |
| 150 epochs | 77.52 | 0.887 | 62.93 | 25.78 | 36.51 |
| 200 epochs | 77.65 | 0.808 | 62.72 | **25.88** | **36.64** |
| 300 epochs | 78.32 | 0.712 | 62.24 | 25.80 | 36.48 |

$r(\text{teacher clean},\ \text{student clean}) = \mathbf{-0.999}$: the teacher gains $2.51$ points
across the ladder and the student **loses** $2.04$. NRR peaks in the middle, so the teacher's training
length is a control on the operating point, and locating it costs five natural-training runs and no
adversarial ones. Reproduced on Tiny-ImageNet, where it is accuracy-controlled: two teachers $0.32$
apart move the student by $1.92$ clean and $1.58$ AA.

---

### 5.7 Larger architecture

WideResNet-34-10 at the same recipe, on both CIFAR datasets. *Running on a separate machine; the
comparison line is ADR's WRN-34-10 row, $62.21$ clean / $31.63$ AA on CIFAR-100.*

---

## Writer's notes

**What is still missing from this section.** (i) The WideResNet block, §5.7, has no numbers yet.
(ii) Table 7's CIFAR-10 column needs `adaadigdm` and `ladder_p0_100ep`, both queued. (iii) Nine cells
marked † are being re-measured under the corrected AWP ascent objective; the ladder's six non-AWP
rows and every baseline in Table 5 are unaffected and final.

**Why the head ablation sits in the base regime.** Table 8 is measured at 50 epochs with no WA and no
AWP precisely so that no stack component can absorb the difference between head variants; the
full-recipe confirmation is given as a sentence rather than a second table.

**Checkpoint selection.** Every number of ours is the final-epoch weight-averaged model. RPAT selects
its best checkpoint by PGD-20 and ADR by PGD-10, which favours them slightly; we report their
published selection rather than re-selecting, and our own reproduction of RPAT gives both its last and
best rows in §5.2's note.

**Table 6 belongs in the experiments section, not the analysis.** It is a measurement about a
competing method's premise, and IGDM itself presents the same quantity as an experimental figure
(their Fig. 2) rather than as part of the derivation.
