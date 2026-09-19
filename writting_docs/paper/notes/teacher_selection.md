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

## One cheap prediction to test next

`clean_cos200ep` is the one off-trajectory teacher already trained here (cosine schedule, 200 epochs):
clean 73.00, S_w/S_b 2.257, margin 0.223, gradient norm 4.33, feature sensitivity 0.722. Every metric
above places it at or below the 5-epoch teacher, while its clean accuracy is that of the 20-epoch one.
The rule therefore predicts a student well below 54.54 clean and 16.00 AA, whereas clean accuracy
predicts about 63/23. One 50-epoch student, about an hour, decides between them.
