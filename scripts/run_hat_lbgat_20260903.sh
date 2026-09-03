#!/usr/bin/env bash
# HAT AND LBGAT, PORTED AND RUN HERE (2026-09-03).
#
# Both consume a naturally trained network, and they do different things with it, which is the reason
# both are worth having as measured rows rather than citations:
#
#   LBGAT  reads it at the CLEAN point and matches its LOGITS under a squared error, no temperature.
#          That is three of our four choices; the fourth is the layer.  It is the sharpest available
#          test of "feature beats logit at the same read point with no temperature on either side",
#          which is a claim the paper currently cannot make from measurement.
#          Its natural branch trains JOINTLY and from RANDOM init -- load: False is deliberate.
#
#   HAT    uses it only to LABEL helper points beyond the ball; none of its representation transfers.
#          It belongs in the trade-off group with ReBAT and RPAT, not with the distillation methods.
#
# Protocol is our baseline one (SGD 0.1, decay 70/90, 100 epochs) so both are comparable with TRADES
# and MART in the same table.  LBGAT's own script is 100 epochs at 76/91, near-identical; HAT's is 50
# adversarial epochs, so ours gives it more rather than less.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
# LBGAT is re-run on both datasets with `lr_schedule: lbgat`, its own schedule, after the first pass
# under our flat-0.1 protocol died on CIFAR-10 -- pinned at 9.9999 clean from step 0 while CIFAR-100
# trained normally under an identical config.  Their adjust_learning_rate carries an epoch-1 dip to
# 0.02 that we had dropped, and reporting the 10.00 would have published our optimizer failure as
# their method.  CIFAR-100's flat-protocol result (57.10 / 25.99) is kept in the notes for comparison.
for c in lbgat_100ep hat_50ep; do
  for ds in CIFAR100 CIFAR10; do
    if [ "$c" != lbgat_100ep ] && ls results/$ds/*/$c/*.log >/dev/null 2>&1 &&        grep -ql "last_aa_acc" results/$ds/*/$c/*.log 2>/dev/null; then
      echo "=== $(date '+%m-%d %H:%M') skip $ds/$c (이미 AA 있음) ==="; continue
    fi
    echo "=== $(date '+%m-%d %H:%M') start $ds/$c ==="
    $PY -u main.py --config_name "${c}.yaml" --dataset "$ds" --seed 0 > "logs/${ds}_${c}.log" 2>&1
    echo "=== $(date '+%m-%d %H:%M') done $ds/$c (exit $?) ==="
  done
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
