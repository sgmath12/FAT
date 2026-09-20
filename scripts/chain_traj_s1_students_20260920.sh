#!/usr/bin/env bash
# 2026-09-20.  Students for the seed-1 teacher ladder, all six checkpoints.  Six rather than the window
# plus neighbours, because "the window contains the best checkpoint" cannot be checked without the best.
# The window is printed by the preceding chain before this one starts a single student, so the log order
# is the record that the rule was applied blind.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
while pgrep -f 'main.py --config_name clean_s1' > /dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') seed-1 teacher ladder finished ==="
for e in 150 200 100 300 40 20; do   # window first, then its neighbours, then the ends
  cfg=tladder_s1_${e}ep
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
