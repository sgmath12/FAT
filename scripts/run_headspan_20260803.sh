#!/usr/bin/env bash
# Queued behind featdir_champ200_trawsnorm. Mechanism test: direction loss restricted to the head
# row space (featdir_span: teacher, k=100). See config header for the hypothesis and both outcomes.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:-929549}"
echo "=== $(date '+%m-%d %H:%M') waiting on pid $WAIT_PID (trawsnorm) ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') start featdir_champ200_headspan ==="
$PY main.py --config_name featdir_champ200_headspan.yaml --dataset CIFAR100 --seed 0 > logs/featdir_champ200_headspan.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
