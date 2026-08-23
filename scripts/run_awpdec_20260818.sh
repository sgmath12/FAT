#!/usr/bin/env bash
# STACK DECOMPOSITION step 3 (2026-08-18): isolate AWP from freeze_lr_epoch.
# Steps 1-2 (C100 / ResNet18 / 100ep / seed 0, direction vs L2, clean / AA):
#   nothing      61.52 / 22.90  vs  62.40 / 24.34   -> L2 +1.44
#   +WA          61.34 / 25.55  vs  61.58 / 26.55   -> L2 +1.00
#   +WA +eps8.8  59.96 / 26.30  vs  60.09 / 27.00   -> L2 +0.70
#   full stack   60.74 / 28.69  vs  57.78 / 28.04   -> dir +0.65
# Neither WA nor eps 8.8 reverses the order; 1.35 of the 2.09 gap swing is in AWP+freeze_lr.
# These two cells add AWP proxy only.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in wadec_dir_wa_eps88_awp wadec_raw_wa_eps88_awp; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
