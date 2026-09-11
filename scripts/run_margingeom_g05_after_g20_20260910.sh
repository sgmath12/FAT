#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

while pgrep -f '[m]ain.py --config_name margingeom_g20.yaml' >/dev/null; do
  sleep 30
done

echo "=== $(date '+%m-%d %H:%M') start CIFAR100/margingeom_g05 ==="
"$PY" -u main.py --config_name margingeom_g05.yaml --dataset CIFAR100 --seed 0 \
  > logs/CIFAR100_margingeom_g05_s0.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done CIFAR100/margingeom_g05 ==="
