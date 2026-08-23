#!/usr/bin/env bash
# ANGULAR TOLERANCE sweep (2026-08-19), BASE regime -- no WA / no AWP / eps 8 / 100ep / angeps off.
# Baseline to beat in this regime: direction 61.52 / AA 22.90, raw L2 62.40 / AA 24.34.
# Beating 24.34 moves the directional design's advantage out of the stack and into the loss.
# Released fraction at the base run's convergence: m 0.85 -> 64%, 0.90 -> 46%, 0.95 -> 17%.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in angtol_090 angtol_085 angtol_095; do   # middle release fraction first
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
