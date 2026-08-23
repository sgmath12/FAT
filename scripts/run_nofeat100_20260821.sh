#!/usr/bin/env bash
# Pure-KD (no feature/head split) design axis at 100ep, no stack.  Direction vs raw L2, tau 16,
# nothing else changed.  50ep versions: direction 61.69 / CW 27.05, raw 58.43 / CW 26.70.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for c in nofeat100_norm nofeat100_raw; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
