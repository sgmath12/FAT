#!/usr/bin/env bash
# gamma=1.0 dropped (trick A monotone-negative at 0.25/0.5). ETF runs right after proto_g025.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:?}"
echo "=== $(date '+%m-%d %H:%M') waiting on proto_g025 pid $WAIT_PID ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') start etf_rotate ==="
$PY main.py --config_name etf_rotate.yaml --dataset CIFAR100 --seed 0 > logs/etf_rotate.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
