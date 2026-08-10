#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for c in beta_b20 beta_b05; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== beta sweep done $(date) ==="
