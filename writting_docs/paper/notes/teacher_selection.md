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

Two different teacher properties predict the two axes, and neither is clean accuracy:

- **Student robustness** follows the teacher's class structure: logit margin 0.98, class separation
  −0.93 (smaller ratio, stronger separation, better student AA). Teacher clean accuracy scores 0.95
  here only because it is itself monotone in training time.
- **Student clean accuracy** follows the teacher's input sensitivity, 0.90 for the gradient norm and
  0.93 for the feature sensitivity, against 0.18 for teacher clean accuracy. Both sensitivities are
  non-monotone along the trajectory: they rise to the 40--50-epoch teacher and fall afterwards, which
  is the same shape as the student clean accuracy the ladder reports, and it is why they track it while
  clean accuracy does not.
- The ratio scores are dominated by the margin term, so they add nothing over the margin alone here.

## What this does and does not establish

All nine points come from one trajectory, so every monotone metric correlates with every other:
teacher clean accuracy has rank correlation 1.00 with both ratio scores and 0.90 with the margin. The
two sensitivities are the exception, correlating 0.43 and 0.32 with teacher clean accuracy, and they
are the metrics that predict a student axis clean accuracy does not. That asymmetry is the finding;
attribution still needs teachers whose geometry differs at matched accuracy.

Also, the shipped teachers were trained on the full training set, so these metrics are read on test
images, the same split the students are scored on. A selection rule claimed from this alone is not
independent of that split.

## One cheap prediction to test next

`clean_cos200ep` is the one off-trajectory teacher already trained here (cosine schedule, 200 epochs):
clean 73.00, S_w/S_b 2.257, margin 0.223, gradient norm 4.33, feature sensitivity 0.722. Every metric
above places it at or below the 5-epoch teacher, while its clean accuracy is that of the 20-epoch one.
The rule therefore predicts a student well below 54.54 clean and 16.00 AA, whereas clean accuracy
predicts about 63/23. One 50-epoch student, about an hour, decides between them.
