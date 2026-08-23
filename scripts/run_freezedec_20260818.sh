#!/usr/bin/env bash
# STACK DECOMPOSITION step 5 (2026-08-18): freeze_lr_epoch WITHOUT AWP.
# The cumulative chain only ever measured freeze_lr on top of AWP.  freeze_lr is the element that
# changes the sign of the direction-vs-L2 gap, and it does so by hurting L2 (clean -2.25, AA -0.01)
# rather than helping direction (AA +0.48, clean -0.79).  These cells decide whether that penalty is
# a property of the frozen-LR tail or an AWP x freeze_lr interaction.
# Reference (C100 / ResNet18 / 100ep / seed 0, direction vs L2, clean / AA):
#   +WA+eps8.8       59.96 / 26.30  vs  60.09 / 27.00
#   +WA+eps8.8+AWP   61.41 / 28.07  vs  60.21 / 28.14
#   full champion    60.62 / 28.55  vs  57.96 / 28.13
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in wadec_raw_wa_eps88_freeze wadec_dir_wa_eps88_freeze; do   # L2 first: it carries the effect
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
