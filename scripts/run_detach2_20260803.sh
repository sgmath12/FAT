#!/usr/bin/env bash
# Detached-head 2-cell: raw L2 vs cosine, everything else identical (champion recipe, alpha 0.0).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for c in featdir_detach_cos featdir_detach_l2; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== detach2 complete $(date) ==="
