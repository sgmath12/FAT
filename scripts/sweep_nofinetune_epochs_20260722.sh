#!/bin/bash
# 2026-07-22: does more training let the finetune:False (random-init) student catch up to the
# finetune:True warm-started one? Same recipe (300ep teacher, train_eps=8/255, lamda=4) at
# epochs in {60,100,150}. Sequential, single GPU.
set -e
cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
DRIVER_LOG=results/CIFAR100/sweep_nofinetune_epochs_20260722_driver.log

for ep in 60 100 150; do
  cfg="featdir_teacher300ep_eps8_lamda4_${ep}ep"
  echo "=== START ${cfg} $(date) ===" >> "$DRIVER_LOG"
  $PY -u main.py --config_name "${cfg}.yaml" --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 >> "$DRIVER_LOG" 2>&1
  echo "=== DONE ${cfg} $(date) ===" >> "$DRIVER_LOG"
done
echo "NOFINETUNE_EPOCH_SWEEP_DONE $(date)" >> "$DRIVER_LOG"
