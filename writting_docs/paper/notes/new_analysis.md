# Moving the paper's centre of gravity to the teacher question (2026-09-19)

## The proposed reframing, and what it costs

Current spine: *a natural teacher's clean features can be the sole target for adversarial training.*

Proposed spine: *using a natural teacher is not enough --- which checkpoint is selected, and how the
perturbation budget is allocated per example, is what sets the student's clean--robust trade-off.*
Feature anchoring becomes the minimal instrument that makes both questions answerable.

    natural teacher  ->  { which checkpoint?  how much perturbation per example? }  ->  student trade-off

This is the better framing for novelty: it stops the paper reading as one more feature-distillation
method, and it answers the reviewer objection that the checkpoint observation gives no selection rule.
The cost is that the new framing promises a rule, and a promise is graded. Today's evidence is nine
checkpoints from one trajectory, where the residual test shows no metric adding information over
teacher accuracy. So the reframing is conditional on the two runs below reproducing the asymmetry; if
they do not, the correlations stay an appendix observation and the current spine stands.

## What the step-1 numbers do and do not say

Corrected reading, replacing an earlier overstatement of ours:

> Teacher clean accuracy is insufficient as a selection criterion: it predicts the robustness ordering
> along this trajectory ($\rho = 0.95$ with student AA) but not the student's clean accuracy ($0.18$).
> Teacher sensitivity and class geometry separately predict the two axes of the student trade-off.

| student axis | strongest teacher predictor | reading |
|---|---|---|
| clean | feature sensitivity, $\rho = 0.93$ | the teacher's local input response |
| AA | logit margin, $\rho = 0.98$ | the teacher's class structure |
| NRR | logit margin, $\rho = 0.98$ | NRR ordering is carried by AA here |
| clean | teacher clean accuracy, $\rho = 0.18$ | not a usable criterion for this axis |

Three caveats that have to travel with those numbers, all measured in
`notes/teacher_selection.md`:

- **Collinearity.** Teacher clean accuracy has rank correlation $1.00$ with the epoch index and $0.90$
  with the margin, so margin's $0.98$ against accuracy's $0.95$ is not evidence of a better criterion.
  The sensitivities are the only metrics off that axis ($0.43$, $0.32$).
- **Weak prediction where it matters.** Leave-one-checkpoint-out error on student clean accuracy is
  $1.32$ points for the gradient norm against $1.90$ for predicting the mean; the rank correlation of
  $0.93$ is not a strong fit, which is why the figure should be scatter plots rather than correlation
  bars.
- **No added information yet.** After regressing the student on teacher accuracy, the residual's
  correlation with the margin is $0.48$ and with feature sensitivity $0.30$: nothing that $n = 9$
  supports. Step 1 is predictor selection, not a test.

## Runs that decide the reframing

**Pre-registered predictions, written before the runs finish.** This is the cheap defence against the
post-hoc reading, and it is the reason to write them here rather than after the fact.

| teacher | clean accuracy | margin | feature sens. | accuracy rule predicts | geometry rule predicts |
|---|---|---|---|---|---|
| `clean_ls01_200ep` (label smoothing 0.1) | expected near 77 | expected lower than 4.48 | to be measured | student near 62.7 / 25.9 | clean follows sensitivity, AA follows the reduced margin: lower AA than the 200-epoch teacher |
| `clean_mixup_200ep` | expected 74--78 | expected much lower | expected lower | student near 62.7 / 25.9 | markedly lower AA |
| `clean_wd5e3_200ep` (weight decay 5e-3) | expected near 76 | expected higher | expected lower | student near 62.7 / 25.9 | AA at or above the 200-epoch teacher's, clean below it |

The three new teachers are trained with `clean_200ep`'s schedule and one change each, so their clean
accuracy should stay within a couple of points of $77.38$ while their geometry moves. Each is followed
by the same 50-epoch ladder student, the only difference from the other nine points being the teacher.

Queued as `scripts/chain_teachersel_20260919.sh` behind the Table 4 chain: three teachers at $0.39$ h
and three students at about $1.05$ h, so roughly $4.5$ h in total.

A fourth candidate was dropped. `clean_cos200ep` looked like a free off-trajectory teacher, but it is
the normalized-feature variant of the network, so the metric script had loaded it with a parameter
silently dropped and its student could not be trained at all; `notes/teacher_selection.md` records the
retraction and the script now checks for it. That leaves the three teachers above, all trained with the
same network as the ladder.

**Decision rule.** If the geometry metrics keep their ordering on teachers whose accuracy is matched,
the Analysis section is retitled *What Predicts Transfer from a Natural Teacher?* and the reframing
goes into the abstract and introduction. If accuracy predicts these four as well as the geometry
metrics do, the correlations move to the appendix as an observation and the paper keeps its current
spine.

**Outcome, 2026-09-20 (details in `notes/teacher_selection.md`).** The rule splits. The clean axis
passes: label smoothing and mixup give teachers $0.14$ points apart in clean accuracy and students
$4.18$ points apart in clean accuracy, ordered by feature sensitivity, which accuracy cannot see at
all. The robust axis fails: the margin and separation rules, strongest on the trajectory, err by $3.05$
and $7.95$ points out of sample where teacher accuracy errs by $0.92$. So the reframing goes ahead in
the halved form --- one teacher property that accuracy misses, for one axis --- and the margin rule is
dropped rather than reported as a criterion.

## Section and figure changes, if the reframing goes ahead

Analysis, in this order: the obvious criterion is teacher accuracy; nine checkpoints show it fails on
student clean accuracy; feature sensitivity predicts that axis; margin and $S_w/S_b$ predict
robustness; therefore two teacher properties, not one quality scalar, describe the trade-off.

\Cref{fig:teacherladder} is replaced by two scatter plots, teacher margin against student AA and
teacher feature sensitivity against student clean accuracy, with each point labelled by teacher epoch
and the off-trajectory teachers drawn with a different marker. The correlation table moves to the
appendix. Scatter plots are the honest presentation at $n = 9 + 4$; correlation bars are not.

## How this connects to the sample-wise radius, and how it must not

The two findings share the word sensitivity and are not the same quantity. Teacher selection uses the
sensitivity of the frozen teacher, measured once per checkpoint; the radius rule uses the per-example
input gradient of the student's anchor loss, measured every step. Neither derives the other, and the
paper should not suggest it does:

> Teacher-level sensitivity helps characterize which representation is transferred, whereas sample-level
> anchor sensitivity determines how strongly each example should be perturbed once the teacher is fixed.

One measurable link is worth stating because it is a fact rather than an analogy: the student is warm
started at the teacher, so at step zero the anchor-loss gradient is evaluated at the teacher's own
weights and the two sensitivities coincide; they separate as training proceeds. Correlating the
teacher's per-example feature sensitivity with the first epoch's allocation multipliers takes minutes
and would let that sentence be quantitative.
