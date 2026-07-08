#!/bin/bash
# FINE positive-gamma sweep for fixed per-class tau (user, 2026-07-07 20:4x): first sweep showed
# gamma +0.25 H 41.58 (~tie w/ baseline 41.77) vs -0.25 H 40.41 (clearly worse) -> negative sign
# dead, probe the small-positive region. Chains behind run_tauclass_fixed_sweep.sh (gamma 0.5, -0.5
# still pending there). Same log/parse convention: temp_tauclass_fixed/output.log, parse by gamma.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
while pgrep -f "run_tauclass_fixed_sweep.sh" > /dev/null || pgrep -f "main.py --config_name temp_tauclass_fixed.yaml" > /dev/null; do sleep 60; done
echo "coarse sweep done, starting fine sweep $(date)"
for g in 0.3 0.15 0.1; do
  echo "=== run: gamma $g $(date) ==="
  $PY -u main.py --config_name temp_tauclass_fixed.yaml --dataset CIFAR100 --seed 0 --gamma $g \
    > results/CIFAR100/tauclass_fixed_driver.log 2>&1
done
echo "TAUCLASS_FIXED_FINE_DONE $(date)"
