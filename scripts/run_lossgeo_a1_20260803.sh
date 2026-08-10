#!/usr/bin/env bash
# Third run: fills the missing cell of the alpha-1.0 2x2 (plain architecture + cosine loss).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:-947011}"
echo "=== $(date '+%m-%d %H:%M') waiting on lossgeo queue pid $WAIT_PID ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start lossgeo_cos_a1 ==="
$PY main.py --config_name lossgeo_cos_a1.yaml --dataset CIFAR100 --seed 0 > logs/lossgeo_cos_a1.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
