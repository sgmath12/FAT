#!/usr/bin/env bash
# The rest of table 5: the methods that do not consume a teacher, given the natural teacher as an
# initialization at their own recipe (waits on the master chain, PID 1703363).
#
# TRADES and MART have never completed in this harness -- TRADES is what surfaced the KLDivLoss NaN
# fixed on 09-07 -- so their numbers have to be read against the published 55.39 / 24.51 and
# 49.83 / 25.00 before the teacher-init rows mean anything.  Hence the own-recipe random-init cells
# run first, as the reproduction check.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1703363
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in trades_100ep mart_100ep trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
