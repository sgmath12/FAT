#!/usr/bin/env bash
# Schedule sensitivity and teacher quality (waits on the table-5 expansion, PID 1698917).
#
# Order is by cost: the four weak teachers are minutes each, the short student runs are under an
# hour, the learning-rate sweep is 2.2 h a cell, and the 200-epoch cell is last because it is 4.4 h
# on its own.  The teacher-quality STUDENTS are not here: which rungs to run depends on the clean
# accuracies these teachers land at, so those configs are written once the teachers report.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1698917
while kill -0 "$PID" 2>/dev/null; do sleep 60; done

for cfg in clean_5ep clean_10ep clean_20ep clean_40ep \
           ship_ep10 ship_ep25 \
           ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 \
           ship_ep200; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
