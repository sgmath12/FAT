#!/usr/bin/env bash
# Causal geometry sweep.  The natural teacher, initialization and inherited classifier are fixed;
# only the diagonal metric used by the matched feature-anchor attack and outer loss changes.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

for cfg in geomdiag_smoke geomdiag_gm05 geomdiag_g05 geomdiag_g10; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  "$PY" -u main.py --config_name "${cfg}.yaml" --dataset CIFAR100 --seed 0 \
    > "logs/CIFAR100_${cfg}_s0.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg ==="
done
