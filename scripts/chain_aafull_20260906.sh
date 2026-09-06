#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1686680
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') FAB-T and Square on the full test set ==="
$PY -u scripts/eval_aa_full_20260906.py > logs/aa_full_20260906.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
