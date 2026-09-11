#!/usr/bin/env bash
# 2026-09-11.  tab:main's second question for the methods that do not consume a teacher: given our
# optimizer, our schedule, our natural warm start and our stack, what do TRADES, MART, HAT and LBGAT
# reach?  The four distillation objectives already have this cell; these complete the row.  The
# learning-rate sweep and the 200-epoch cell follow, since they were the standing queue.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=16874
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in trades_full_100ep mart_full_100ep hat_full_100ep lbgat_full_100ep \
           ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 ship_ep200; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
