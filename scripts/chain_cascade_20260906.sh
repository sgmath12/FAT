#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1686680
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') running the AA cascade check ==="
$PY -u scripts/eval_aa_cascade_20260906.py > logs/aa_cascade_20260906.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
