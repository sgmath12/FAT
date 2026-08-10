#!/usr/bin/env bash
# Trick B standalone, queued behind the proto sweep.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:?need proto queue pid}"
echo "=== $(date '+%m-%d %H:%M') waiting on proto queue pid $WAIT_PID ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start etf_rotate ==="
$PY main.py --config_name etf_rotate.yaml --dataset CIFAR100 --seed 0 > logs/etf_rotate.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
