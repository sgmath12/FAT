#!/usr/bin/env bash
# Trick A: class-prototype shrinkage sweep. g=0 already exists (fg_plain_th_sh_kl 62.61/29.16/26.63).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for c in proto_g05 proto_g025 proto_g10; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== proto sweep complete $(date) ==="
