#!/usr/bin/env bash
# Pure loss-geometry pair: plain architecture in both cells (reformation/student_norm/teacher_norm
# False), head detached (alpha 0), only the feature distance differs -- cosine vs raw L2.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for c in lossgeo_cos lossgeo_l2; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== lossgeo complete $(date) ==="
