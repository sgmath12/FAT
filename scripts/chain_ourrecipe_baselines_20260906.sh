#!/usr/bin/env bash
# Chained after run_abl_schedule_20260906.sh (PID 1683338 at write time).  Waits on the PID rather than
# racing the GPU flock, which starved a waiter for 393 minutes on 09-04.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1683338
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') schedule ablation finished, starting the our-recipe baselines ==="
for cfg in adaad_ourrecipe_100ep rslad_ourrecipe_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
