#!/usr/bin/env bash
# Table 5 expansion: natural-teacher INITIALIZATION at each method's own recipe (batch 1), then our
# optimizer and schedule without the stack (batch 2).  Waits on the tau sweep (PID 1694030).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1694030
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in ard_natinit_100ep rslad_natinit_100ep adaad_natinit_100ep adaadigdm_natinit_100ep \
           ard_ourrecipe_100ep adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
