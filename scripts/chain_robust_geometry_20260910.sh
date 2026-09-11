#!/usr/bin/env bash
# Teacher-only robustness-aware geometry sweep.  The inherited head and clean feature target are
# fixed; only the metric on their feature discrepancy changes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

for cfg in robgeom_smoke robgeom_g05 robgeom_g10; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  "$PY" -u main.py --config_name "${cfg}.yaml" --dataset CIFAR100 --seed 0 \
    > "logs/CIFAR100_${cfg}_s0.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg ==="
done
