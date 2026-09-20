# Can a teacher be chosen without training a student? (step 1, 2026-09-19)

Teacher-only metrics on the nine CIFAR-100 natural checkpoints of the ladder, measured on 5000 held-out
images with `scripts/teacher_metrics.py`, ranked against the 50-epoch anchor students those same
checkpoints produced (`scripts/teacher_metrics_correlate.py`).  No student was trained for this.

| teacher | clean | S_w/S_b | margin | ‖∇x CE‖ | feat. sens. | margin/∇ | student clean | student AA |
|---|---|---|---|---|---|---|---|---|
| 5 ep | 56.54 | 1.337 | 0.61 | 9.46 | 2.46 | 0.064 | 54.54 | 16.00 |
| 10 ep | 68.06 | 1.170 | 1.77 | 13.08 | 3.21 | 0.136 | 61.04 | 19.94 |
| 20 ep | 73.20 | 1.104 | 2.94 | 17.78 | 4.05 | 0.165 | 63.40 | 22.98 |
| 40 ep | 75.26 | 1.092 | 3.95 | 20.92 | 4.75 | 0.189 | 63.79 | 24.23 |
| 50 ep | 75.32 | 1.093 | 4.10 | 21.04 | 4.69 | 0.195 | **64.07** | 24.51 |
| 100 ep | 76.38 | 0.978 | 4.43 | 19.64 | 4.53 | 0.225 | 63.65 | 25.20 |
| 150 ep | 76.68 | 0.889 | 4.47 | 19.05 | 4.35 | 0.235 | 62.93 | 25.78 |
| 200 ep | 77.38 | 0.811 | 4.48 | 18.50 | 4.14 | 0.242 | 62.72 | **25.88** |
| 300 ep | 77.92 | 0.713 | 4.39 | 18.05 | 3.83 | 0.243 | 61.85 | 25.59 |

## Spearman rank correlation (n = 9)

| teacher metric | student clean | student AA | student NRR |
|---|---|---|---|
| clean accuracy | **0.18** | 0.95 | 0.95 |
| S_w/S_b | −0.17 | −0.93 | −0.93 |
| logit margin | 0.30 | **0.98** | **0.98** |
| ‖∇x CE‖ | **0.90** | 0.48 | 0.48 |
| feature sensitivity | **0.93** | 0.40 | 0.40 |
| margin / ‖∇x CE‖ | 0.18 | 0.95 | 0.95 |
| margin / feature sensitivity | 0.18 | 0.95 | 0.95 |

Teacher clean accuracy is not useless, it is insufficient: it predicts the robustness ordering along
this trajectory ($\rho = 0.95$ with student AA) but not the student's clean accuracy ($0.18$). The two
axes are tracked by different teacher properties:

- **Student robustness** follows the teacher's class structure: logit margin 0.98, class separation
  −0.93 (smaller ratio, stronger separation, better student AA).
- **Student clean accuracy** follows the teacher's input sensitivity, 0.90 for the gradient norm and
  0.93 for the feature sensitivity. Both are non-monotone along the trajectory, rising to the
  40--50-epoch teacher and falling afterwards, which is the shape of the student clean accuracy the
  ladder reports; teacher clean accuracy rises throughout and cannot track it.
- The ratio scores are dominated by the margin term and add nothing over the margin alone. Forcing the
  two axes into one scalar loses the part that is specific to clean accuracy.

## Collinearity: what moves together along one trajectory

Spearman between the teacher metrics themselves.

| | epoch | clean | margin | S_w/S_b | ‖∇x‖ | feat. sens. |
|---|---|---|---|---|---|---|
| epoch | 1.00 | **1.00** | 0.90 | −0.98 | 0.43 | 0.32 |
| clean | 1.00 | 1.00 | **0.90** | **−0.98** | 0.43 | 0.32 |
| margin | 0.90 | 0.90 | 1.00 | −0.88 | 0.53 | 0.47 |
| S_w/S_b | −0.98 | −0.98 | −0.88 | 1.00 | −0.42 | −0.33 |
| ‖∇x‖ | 0.43 | 0.43 | 0.53 | −0.42 | 1.00 | **0.97** |
| feat. sens. | 0.32 | 0.32 | 0.47 | −0.33 | 0.97 | 1.00 |

Teacher clean accuracy has rank correlation $1.00$ with the epoch index and $0.90$ with the margin, so
margin's $0.98$ against accuracy's $0.95$ on student AA cannot be read as margin being the better
criterion. The two sensitivities are the only metrics that are not collinear with training time
($0.43$, $0.32$), and they are the ones that predict the axis accuracy misses.

## Leave-one-checkpoint-out prediction

A line fit on eight checkpoints, evaluated on the ninth; mean absolute error in accuracy points.

| predictor | student clean | student AA |
|---|---|---|
| feature sensitivity | 1.37 | 1.85 |
| ‖∇x CE‖ | **1.32** | 1.52 |
| logit margin | 2.08 | **0.56** |
| S_w/S_b | 2.77 | 2.09 |
| teacher clean accuracy | 1.72 | 0.78 |
| epoch index | 2.59 | 2.75 |
| predicting the mean | 1.90 | 2.47 |

Student AA is genuinely predictable: the margin errs by $0.56$ points against $2.47$ for the mean.
Student clean accuracy is much harder, and the sensitivities are the best of a weak field, $1.32$
against $1.72$ for teacher accuracy and $1.90$ for predicting the mean.

## Does any metric add information beyond teacher accuracy?

Regressing the student on teacher clean accuracy and rank-correlating the residual with each metric:

| teacher metric | student clean residual | student AA residual |
|---|---|---|
| feature sensitivity | 0.30 | 0.05 |
| ‖∇x CE‖ | 0.20 | 0.12 |
| logit margin | −0.43 | 0.48 |
| S_w/S_b | 0.60 | −0.28 |
| epoch | −0.57 | 0.32 |

Nothing survives this at $n = 9$. The largest values, $0.60$ for $S_w/S_b$ on the clean residual and
$0.48$ for the margin on the AA residual, are what nine points give by chance often enough that they
cannot be claimed. So the honest reading of step 1 is a hypothesis with one supporting asymmetry:
sensitivity is the only teacher quantity that is not a restatement of training progress, and it is the
only one that predicts student clean accuracy at all. Whether it adds information over accuracy needs
teachers whose geometry differs at matched accuracy, which is step 1.5, and new trajectories, step 2.
## What this does and does not establish

All nine points come from one trajectory, and the tables above are a hypothesis-generating step rather
than a test: no p-value is claimed, the predictor set was compared after the students were known, and
the residual analysis shows no metric adding to teacher accuracy at this sample size. What step 1
establishes is which predictors are worth carrying forward, and that a single scalar score is the wrong
target: the two axes want different teacher quantities.

Also, the shipped teachers were trained on the full training set, so these metrics are read on test
images, the same split the students are scored on. A selection rule claimed from this alone is not
independent of that split.

## A retracted prediction

An earlier version of this note proposed `clean_cos200ep` as an off-trajectory teacher whose geometry
contradicted its accuracy, and predicted a student from it. That measurement was invalid. The
checkpoint is the normalized-feature variant of the network: it carries `log_s` where this ResNet-18
carries `alphas`, and the metric script loaded it with `strict=False`, dropping that parameter. The
margin of $0.223$ and feature sensitivity of $0.722$ it reported are artifacts of the dropped feature
scale, and the ladder student for it cannot run at all --- the state dict does not fit the plain
network, which is how the mistake surfaced. The script now asserts that a checkpoint carries no
parameters the network lacks.

The accuracy-matched teachers of `notes/new_analysis.md` (label smoothing, mixup, weight decay) are
trained with this same network and replace it as the off-trajectory test.

# Step 1.5: teachers whose geometry differs at matched accuracy (2026-09-20)

Three teachers trained with `clean_200ep`'s schedule and one change each, then the same 50-epoch ladder
student. Metrics as above, on 5000 held-out images.

| teacher | clean | S_w/S_b | margin | ‖∇x‖ | feat. sens. | student clean | student AA |
|---|---|---|---|---|---|---|---|
| 200 ep (trajectory) | 77.38 | 0.811 | 4.48 | 18.50 | 4.14 | 62.72 | 25.88 |
| label smoothing 0.1 | **78.76** | 0.679 | 3.08 | 17.23 | **3.17** | **59.62** | 25.72 |
| mixup | **78.62** | 1.680 | 2.12 | 20.15 | **4.81** | **63.80** | 25.70 |
| weight decay 5e-3 | 69.30 | 0.640 | 1.70 | 19.68 | 4.13 | 55.30 | 19.60 |

## The accuracy-matched pair decides the clean axis

Label smoothing and mixup land $0.14$ points apart in teacher clean accuracy, $78.76$ against $78.62$,
and produce students $4.18$ points apart in clean accuracy, $59.62$ against $63.80$. Their AutoAttack
accuracies are the same to within $0.02$. So at matched teacher accuracy the clean axis moves and the
robust axis does not, which is the asymmetry step 1 predicted, and teacher accuracy cannot see it: a
line fit on the nine trajectory checkpoints gives $64.3$ and $64.2$ for these two teachers, while the
feature-sensitivity fit gives $59.0$ and $65.0$ against the measured $59.62$ and $63.80$.

The direction is the one step 1 measured. The teacher with the lower feature sensitivity ($3.17$) gives
the lower student clean accuracy ($59.62$), and the more sensitive teacher ($4.81$) the higher ($63.80$).

## The margin rule does not survive the same test

Out-of-sample error on the three new teachers, from lines fit on the nine trajectory checkpoints:

| predicted axis | predictor | MAE (points) | per teacher |
|---|---|---|---|
| student clean | feature sensitivity | **2.99** | 0.7 / 1.2 / 7.2 |
| student clean | ‖∇x CE‖ | 3.51 | 2.2 / 0.1 / 8.3 |
| student clean | teacher clean accuracy | 3.45 | 4.7 / 0.4 / 5.2 |
| student AA | logit margin | 3.05 | 3.3 / 5.5 / 0.4 |
| student AA | S_w/S_b | 7.95 | 2.6 / 12.0 / 9.3 |
| student AA | teacher clean accuracy | **0.92** | 0.4 / 0.4 / 2.0 |

The margin predicted $22.5$ and $20.2$ AutoAttack points for the label-smoothing and mixup teachers,
which actually gave $25.72$ and $25.70$; class separation was worse still, erring by $12.0$ on the
mixup teacher. Their $0.98$ and $-0.93$ rank correlations on the trajectory were the collinearity with
training progress that the correlation matrix flagged, not a criterion. **Teacher clean accuracy is the
better predictor of the robust axis out-of-sample**, erring by $0.92$ points.

The weight-decay teacher is the hard case for every clean-axis rule: at $69.30$ teacher clean accuracy
with sensitivity $4.13$ it gives a $55.30 / 19.60$ student, and the sensitivity fit errs by $7.2$ points
there. Sensitivity wins the matched pair, not the general regression.

## Which sensitivity, and why it matters that it is the feature one

The metric that ordered the matched pair with errors of $0.7$ and $1.2$ points is the **feature
sensitivity**, $\lVert \Phi_t(x+\delta) - \Phi_t(x) \rVert_2 / \lVert \delta \rVert_2$ for uniform
$\delta$ at $8/255$. The CE gradient norm errs by $2.2$ and $0.1$ on the same pair: it happens to nail
the mixup teacher and miss the label-smoothing one. The paper should name the feature version and keep
the gradient version in the appendix, for a reason beyond the smaller error: label smoothing changes
the logit scale and therefore the CE gradient directly, so a label-dependent metric is the wrong thing
to read on exactly this pair, whereas the feature sensitivity uses no labels and no logits.

## What can now be claimed, and what cannot

Supported:

- Two teachers at the same clean accuracy ($78.76$ and $78.62$) give students $4.18$ points apart in
  clean accuracy and equal to $0.02$ in AutoAttack accuracy. Teacher clean accuracy therefore cannot
  select for the student's clean axis, and this is a direct test rather than a correlation.
- Along one trajectory teacher clean accuracy tracks the student's robustness ordering ($\rho = 0.95$).
  Extending to teachers trained differently, that relation does not survive as a selection rule: the
  margin and separation metrics that looked strongest on the trajectory err by $3.05$ and $7.95$ points
  out of sample, and teacher accuracy's own out-of-sample error on the robust axis is $0.92$.
- Feature sensitivity explains the matched pair's clean-accuracy gap better than any other metric here.

Not supported, and not to be written:

- That feature sensitivity selects teachers in general. The weight-decay teacher is a counterexample:
  at clean accuracy $69.30$ with sensitivity $4.13$ it gives $55.30 / 19.60$, and every clean-axis rule
  errs by $5$ to $8$ points on it. This is not a small caveat, it is a refutation of the general rule.
- That the margin or the class separation is a selection criterion for robustness.
- That teacher geometry is the *cause* of either effect. The teacher changes the student's
  initialization, its classifier and its target at once.

The defensible summary is that transfer varies substantially between teachers of equal accuracy, and
that feature sensitivity is the leading candidate for explaining that variation, on $n = 12$
checkpoints, one architecture, one dataset and a single student seed.

## Which sensitivity, verified (2026-09-20)

Two questions had to be settled before the paper names a metric: which of the two sensitivities orders
the matched pair, and whether that ordering is a feature-norm artifact. Both were measured with four
random $\delta$ draws per image rather than one, adding the feature norm and the norm-relative
sensitivity to the metric set.

| teacher | clean | feat. sens. | feat. norm | sens. / norm | ‖∇x CE‖ | student clean |
|---|---|---|---|---|---|---|
| label smoothing 0.1 | 78.76 | 3.16 | 8.68 | 0.366 | 17.23 | 59.62 |
| mixup | 78.62 | 4.82 | 9.53 | 0.507 | 20.15 | 63.80 |
| weight decay 5e-3 | 69.30 | 4.14 | 12.14 | 0.349 | 19.68 | 55.30 |

Prediction error on the matched pair, from lines fit on the nine trajectory checkpoints:

| predictor | label smoothing | mixup | weight decay |
|---|---|---|---|
| feature sensitivity | **−0.71** | **+1.18** | +7.19 |
| ‖∇x CE‖ | +2.18 | +0.10 | +8.26 |
| feature norm alone | +7.46 | +2.18 | +7.28 |
| sensitivity / feature norm | +3.73 | +4.71 | +7.42 |
| teacher clean accuracy | +4.67 | +0.43 | +5.25 |

**It is the absolute feature sensitivity**, $\lVert \Phi_t(x+\delta) - \Phi_t(x) \rVert_2 / \lVert
\delta \rVert_2$, which uses no labels and no logits. The CE gradient norm gets the mixup teacher right
($+0.10$) and the label-smoothing teacher wrong ($+2.18$), and it is the metric label smoothing has a
direct mechanical effect on, so it is the wrong one to name here.

**The ordering is not a norm artifact.** Feature norms differ by a factor $1.10$ across the pair while
the sensitivities differ by $1.52$, and the norm on its own errs by $7.46$ points on the label-smoothing
teacher. Dividing the sensitivity by the norm keeps the pair's ordering ($0.366 < 0.507$) but loses the
calibration, over-predicting both by $3.7$ and $4.7$: along the trajectory the norm-relative sensitivity
is nearly flat against student clean accuracy ($\rho = 0.22$) and tracks AA instead ($0.98$). So the
pair's ordering survives normalization, while the trajectory fit does not, and the honest statement is
about the absolute quantity with that caveat attached.

The weight-decay teacher stays the counterexample under every variant, $+7.2$ to $+8.3$ points.

## Reconciling the notes with the paper's table (2026-09-20)

Two mismatches, both now fixed, and neither in the paper's favour or against it.

- **Teacher clean accuracy.** The first pass measured on a 5000-image subset of the test set and read
  $0.3$ to $0.9$ points low ($75.32$ against the paper's $75.81$ at 50 epochs, and so on). Measured on
  all 10000 images the numbers reproduce `tab:teacherladder_values` exactly: $57.26$, $68.81$, $73.74$,
  $75.74$, $75.81$, $76.62$, $77.52$, $77.65$, $78.32$, and $S_w/S_b$ to within $0.001$. All metrics in
  this note are now the full-test-set version.
- **Student numbers.** Three ladder cells have logs for seeds 0, 1 and 2, and the first pass read the
  newest file, which is seed 2. The paper reports seed 0 throughout, so the correlations were computed
  against a mixture. Corrected to seed 0: $64.28/24.38$ at 50 epochs, $61.15/20.22$ at 10, $62.24/25.80$
  at 300.
- The repeats are useful in themselves: across seeds the same teacher gives students within $0.2$ to
  $0.5$ clean points and $0.1$ to $0.5$ AA points. The matched-pair gap of $4.18$ clean points is an
  order of magnitude above that spread.

## The candidate-window rule, frozen before validation

The goal is not to name one teacher but to narrow a trajectory to the checkpoints worth training
students on. `scripts/teacher_window.py` fixes the rule:

    keep a checkpoint if  margin >= 0.99 * max(margin)  and  feature sensitivity >= 0.85 * max(sens.)

On the original trajectory it keeps exactly the 150- and 200-epoch checkpoints:

| checkpoint | margin | % of max | sensitivity | % of max | kept |
|---|---|---|---|---|---|
| 5 ep | 0.641 | 14.1 | 2.452 | 51.7 | |
| 10 ep | 1.812 | 39.8 | 3.215 | 67.8 | |
| 20 ep | 3.027 | 66.5 | 4.036 | 85.1 | |
| 40 ep | 4.048 | 88.9 | 4.741 | 100.0 | |
| 50 ep | 4.187 | 92.0 | 4.685 | 98.8 | |
| 100 ep | 4.480 | 98.4 | 4.520 | 95.3 | |
| 150 ep | 4.553 | 100.0 | 4.326 | 91.2 | **yes** |
| 200 ep | 4.545 | 99.8 | 4.102 | 86.5 | **yes** |
| 300 ep | 4.473 | 98.2 | 3.832 | 80.8 | |

Those two checkpoints contain the trajectory's best student AutoAttack accuracy ($25.88$ at 200 epochs)
and its best NRR ($36.64$, also 200 epochs), and both give more student clean accuracy than the
300-epoch teacher. The clean-accuracy optimum, the 50-epoch teacher at $64.28$, is outside the window,
which is the cost of the rule and has to be reported as such.

Both thresholds were read off this trajectory after its students were known. They are therefore frozen
in the script and applied unchanged to the two new trajectories now training
(`scripts/chain_teachertraj_20260920.sh`, seeds 1 and 2, snapshots at 20, 40, 100, 150, 200 and 300
epochs from a single run each, using the new `save_epochs` option in `main.py`).

**Protocol for the validation, in order.** Train the two trajectories. Compute teacher metrics on each.
Write the selected window for each seed into this note. Only then train students at the window and at
the checkpoints immediately before and after it, and report whether the window contains that seed's best
student and, if not, by how much clean and AA are given up. Teachers whose accuracy differs by several
points, such as the weight-decay one, are not part of this test; they belong to the separate question of
whether sensitivity ranks teachers across training methods.

## The snapshot shortcut does not work here (2026-09-20)

To get a second trajectory cheaply, `main.py` gained `save_epochs`, writing epoch-indexed snapshots
inside one 300-epoch run. The snapshots measured:

| snapshot | clean | margin | feat. sens. |
|---|---|---|---|
| 20 ep | 62.36 | 1.43 | 4.61 |
| 40 ep | 63.06 | 1.97 | 4.31 |
| 100 ep | 60.43 | 1.79 | 3.83 |
| 150 ep | 60.94 | 1.98 | 4.55 |
| 200 ep | 59.54 | 1.94 | 4.05 |
| 300 ep (final) | 78.27 | 4.52 | 3.84 |

The first five are not N-epoch teachers. Natural training here uses a one-cycle schedule, so a snapshot
taken partway through a 300-epoch run is a model at the schedule's high-learning-rate middle, sitting at
$59$ to $63\%$ clean accuracy on its way to $78.27\%$. The seed-0 ladder was nine separate runs, each
converged under its own one-cycle schedule, and only the final snapshot here is comparable to anything
($78.27$ against seed 0's $78.32$ at 300 epochs, which is a useful check that the seeds agree at the
end). The frozen window rule returns an empty window on this trajectory, correctly: no snapshot has a
saturated margin except the last, whose sensitivity has already fallen.

So the seed-1 ladder is being built the same way as seed 0: one converged run per epoch count, at 20,
40, 100, 150, 200 and 300 epochs, $810$ epochs in total and about $1.6$ h
(`scripts/chain_traj_s1_runs_20260920.sh`). `save_epochs` stays in `main.py` --- it is harmless when
unset and useful for a constant-learning-rate schedule --- but it is not how this trajectory is built.
