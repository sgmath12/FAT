#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:?}"
echo "=== $(date '+%m-%d %H:%M') waiting on angeps_p05 queue pid $WAIT_PID ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start featdir_champ200_angeps ==="
$PY main.py --config_name featdir_champ200_angeps.yaml --dataset CIFAR100 --seed 0 > logs/featdir_champ200_angeps.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
