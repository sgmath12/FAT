# Why does the sensitivity-matched radius move clean accuracy and not robustness? (2026-09-24)

The first-order argument motivates equalizing the perturbation effect across examples. It predicts
nothing about which accuracy axis moves, so the measured asymmetry --- $+2.19$ clean, $+0.03$ AA over
three paired seeds --- is an empirical outcome and has to be written as one. This note records an attempt
to go further, and the attempt half fails.

## The hypothesis under test

A uniform radius may over-constrain the examples whose anchor loss responds strongly to the input: the
same pixel budget forces a larger feature displacement on them, and therefore a stronger invariance
constraint. Sensitivity matching shrinks their radius and moves the budget to less responsive examples.
If that is what happens, the examples receiving smaller radii should show both a smaller clean anchor
error and a larger clean-accuracy gain, with robustness roughly unchanged.

## Measurement

`scripts/radius_decompose.py`, on the final CIFAR-100 pair (150 epochs, $8/255$, WA and AWP, seed 0),
2500 test images in four bins. Per bin: the multiplier the allocation rule assigns, the clean anchor
error $\lVert \Phi_s(x) - \Phi_t(x) \rVert$, the adversarial anchor error under an anchor-loss PGD, clean
accuracy, and PGD-20 accuracy.

### Split A: teacher feature sensitivity, independent of both arms

Bin 0 = least sensitive. This is the only split that does not use either model under comparison.

| bin | multiplier (matched) | clean acc, uniform → matched | clean anchor error, uniform → matched | PGD-20, uniform → matched |
|---|---|---|---|---|
| 0 | 1.060 | 73.76 → 77.12 (**+3.36**) | 5.505 → 5.276 | 40.64 → 41.28 |
| 1 | 1.014 | 61.28 → 65.28 (**+4.00**) | 5.961 → 5.742 | 31.52 → 30.08 |
| 2 | 0.996 | 55.68 → 57.60 (+1.92) | 6.067 → 5.895 | 26.88 → 26.56 |
| 3 | 0.930 | 56.80 → 57.60 (+0.80) | 6.585 → 6.379 | 25.92 → 25.76 |

**The hypothesis fails on this split.** The clean gain is largest where the radius barely changed and
smallest in the bin that received the smallest radius, the reverse of the prediction. The reason is
visible in the multiplier column: teacher feature sensitivity spans only $1.06$ to $0.93$ of the
allocation, while the multipliers actually used during training range over $[0.44, 1.38]$ with a standard
deviation of $0.33$. Teacher sensitivity is a weak proxy for who received a smaller radius.

### Split B: the allocation multiplier the matched model itself assigns

Bin 0 = largest radius, bin 3 = smallest. This is the quantity of interest, at the cost of a split
defined by one of the two arms.

| bin | multiplier (matched) | clean acc, uniform → matched | clean anchor error | adversarial anchor error | PGD-20 |
|---|---|---|---|---|---|
| 0 | 1.339 | 73.62 → 74.67 (+1.04) | 4.582 → 4.533 | 5.799 → 5.726 | 52.31 → 52.91 |
| 1 | 1.256 | 60.97 → 63.73 (+2.76) | 5.218 → 5.150 | 6.499 → 6.548 | 41.28 → 39.21 |
| 2 | 0.877 | 49.60 → 50.56 (+0.96) | 6.413 → 6.240 | 7.975 → 8.142 | 18.08 → 17.76 |
| 3 | 0.522 | 62.40 → 67.84 (**+5.44**) | 7.952 → **7.413** | 9.811 → 9.957 | 12.48 → 12.80 |

**On this split the pattern the hypothesis predicts does appear.** The bin given the smallest radii,
$0.52$, has the largest fall in clean anchor error, $-0.54$ against $-0.05$ in bin 0, and the largest
clean-accuracy gain, $+5.44$. Its robust accuracy is unchanged, $+0.32$, while its adversarial anchor
error rises slightly, $+0.15$. Across all bins the matched model fits the clean target better and the
adversarial target slightly worse.

## What to write, and what not to

The two splits disagree, and the one that supports the hypothesis is the one defined by the model being
tested. That is a selection concern, not a technicality: bin 3 of split B is by construction the set of
examples the matched model chose to perturb least, and those are also the examples whose anchor gradient
was largest at the end of training, which is not independent of how well it fits them. So this is
suggestive and no more.

Two things are safe to state, and are worth stating:

- The role separation. The first-order analysis motivates why a per-example radius is reasonable; the
  clean gain at unchanged AA is what the allocation produced. The analysis does not predict the
  asymmetry, and the paper should say so rather than leave a reader to notice it.
- The direction of the trade at fixed mean budget, which is measurable on both splits: the allocation
  lowers the clean anchor error in every bin and raises the adversarial anchor error slightly in most.
  The same clean-versus-worst-case trade the paper studies at the level of accuracy is visible one level
  down, in the objective itself.

Not to be written: that sensitive examples are over-regularized by a uniform radius and that relieving
them is why clean accuracy rises. Split A refutes the ranking that story needs, and split B cannot
establish it alone. If a reviewer asks for a mechanism, this note is the honest answer: the hypothesis
is natural, one split is consistent with it, an arm-independent split is not, and a proper test needs a
split defined without reference to either model.
