#!/bin/bash
# Gain-only head + feature-direction distillation chain (user, 2026-07-13).
# Order = isolation-first: (1) temp_gainhead = head change only, KL pipeline (direct bar 41.77 /
# coshead 40.01); (2) featdir = pipeline change only, free head; (3) featdir_gainhead = full
# decomposition (teacher head deleted, student head = 100 gains). All 3-step tau16 seed0, ~30min
# each. Registered prediction (memory 2026-07-13): (1) tie/small-loss; (2)(3) open cells.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/gainhead_featdir_chain.log

for cfg in temp_gainhead featdir featdir_gainhead; do
  echo "=== $cfg START $(date) ===" >> $LOG
  $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 \
    > results/CIFAR100/${cfg}_driver.log 2>&1
  echo "=== $cfg DONE $(date) ===" >> $LOG
done
echo "GAINHEAD_FEATDIR_DONE $(date)" >> $LOG
