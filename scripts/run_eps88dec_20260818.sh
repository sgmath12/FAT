#!/usr/bin/env bash
# STACK DECOMPOSITION step 2 (2026-08-18): isolate train_eps 8.8/255.
# Step 1 result (C100 / ResNet18 / 100ep / seed 0, direction vs L2, clean / AA):
#   no WA      61.52 / 22.90  vs  62.40 / 24.34   -> L2 +1.44
#   WA only    61.34 / 25.55  vs  61.58 / 26.55   -> L2 +1.00   <- WA does NOT flip
#   full stack 60.74 / 28.69  vs  57.78 / 28.04   -> direction +0.65
# WA pays both designs nearly equally (+2.65 / +2.21); the flip comes from AWP + eps 8.8 + freeze_lr,
# which buy direction +3.14 AA and L2 only +1.49.  These two cells add ONLY eps 8.8 to WA.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in wadec_dir_wa_eps88 wadec_raw_wa_eps88; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
