#!/usr/bin/env bash
# 2026-09-19.  Step 1.5 of the teacher-selection analysis: three natural teachers whose geometry differs
# at comparable clean accuracy (label smoothing 0.1, mixup, weight decay 5e-3), each followed by the
# same 50-epoch ladder student, so the correlations of notes/teacher_selection.md can be tested off the
# single trajectory they were measured on.  Runs after the Table 4 chain.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
while pgrep -f 'main.py --config_name' > /dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') table-4 chain finished ==="
for cfg in clean_ls01_200ep clean_mixup_200ep clean_wd5e3_200ep \
           tladder_clean_ls01_200ep tladder_clean_mixup_200ep tladder_clean_wd5e3_200ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
