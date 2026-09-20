#!/usr/bin/env bash
# 2026-09-20.  Second priority: the radius-allocation control at the FINAL recipe.  tab:allocation
# compares uniform against sensitivity at 100 epochs and 8.8/255; the main result is 150 epochs at
# 8/255, so this runs the uniform-radius twin of that cell, then repeats both at student seed 1 so the
# AA difference can be read against a seed spread rather than asserted.
# Waits for the teacher-selection students, which are the higher priority.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=94309
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') teacher-selection students finished ==="
run() { echo "=== $(date '+%m-%d %H:%M') start $1 seed $2 ==="; \
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed $2 > logs/CIFAR100_$1_s$2.log 2>&1; \
  echo "=== $(date '+%m-%d %H:%M') done $1 seed $2 (exit $?) ==="; }
run cfa_eps8_150ep_p0 0
run cfa_eps8_150ep 1
run cfa_eps8_150ep_p0 1
