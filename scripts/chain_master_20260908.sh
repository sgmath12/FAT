#!/usr/bin/env bash
# Reordered queue (2026-09-08).  The two comparisons that answer "could a different KD setup do
# this?" go first, because they are what the analysis section's claim rests on:
#   1. conventional KD -- the same temperature on teacher AND student, tau^2 scaled, attack from the
#      same objective.  Under our warm start its clean discrepancy is zero at every tau, which the
#      teacher-only softening cannot say, so it separates "the feature is the better target" from
#      "asymmetric temperature handed the logit target an initial mismatch".
#   2. frozen-head logit MSE -- the same question without a softmax anywhere, swept over three
#      learning rates because the logit and feature losses have different gradient scales.
# Then the rest of the table-5 expansion, then the schedule-sensitivity and weak-teacher cells.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1702965
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in kdsym_t1 kdsym_t2 kdsym_t4 kdsym_t8 kdsym_t16 \
           logitmse_fh_lr0p021 logitmse_fh_lr0p007 logitmse_fh_lr0p042 \
           adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep \
           clean_5ep clean_10ep clean_20ep clean_40ep \
           ship_ep10 ship_ep25 \
           ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 \
           ship_ep200; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
