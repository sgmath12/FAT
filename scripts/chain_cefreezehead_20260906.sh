#!/usr/bin/env bash
# The head control, after the gradient-masking checks (PID 1683820 at write time).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1683820
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start CIFAR100/abl_ce_freezehead_100ep ==="
$PY -u main.py --config_name abl_ce_freezehead_100ep.yaml --dataset CIFAR100 --seed 0 \
  > logs/CIFAR100_abl_ce_freezehead_100ep.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
