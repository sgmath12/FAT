#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:?}"
echo "=== $(date '+%m-%d %H:%M') waiting on noawp_angeps pid $WAIT_PID ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start CIFAR10 champ+angeps ==="
$PY main.py --config_name featdir_champ200_angeps.yaml --dataset CIFAR10 --seed 0 > logs/c10_champ_angeps.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
