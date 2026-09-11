#!/usr/bin/env bash
# Direct intervention on the margin/rotation quantity identified by the teacher-epoch ladder.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

for cfg in margingeom_smoke margingeom_g10; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  "$PY" -u main.py --config_name "${cfg}.yaml" --dataset CIFAR100 --seed 0 \
    > "logs/CIFAR100_${cfg}_s0.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg ==="
done
