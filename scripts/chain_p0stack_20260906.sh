#!/usr/bin/env bash
# The anchor with the stack on a uniform radius -- the rung the ladder never had (waits on PID 1688550).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1688550
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start CIFAR100/ladder_p0_wa_awp_fh_100ep ==="
$PY -u main.py --config_name ladder_p0_wa_awp_fh_100ep.yaml --dataset CIFAR100 --seed 0 \
  > logs/CIFAR100_ladder_p0_wa_awp_fh_100ep.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
