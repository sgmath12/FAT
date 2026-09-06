#!/usr/bin/env bash
# AdaAD with our recipe AND our stack -- the last cell of the 2x2 (waits on PID 1688096).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1688096
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start CIFAR100/adaad_ourrecipe_wa_awp_100ep ==="
$PY -u main.py --config_name adaad_ourrecipe_wa_awp_100ep.yaml --dataset CIFAR100 --seed 0 \
  > logs/CIFAR100_adaad_ourrecipe_wa_awp_100ep.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
