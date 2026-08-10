#!/bin/bash
# 2026-07-22: 4th eps group (train_eps=8/255, i.e. no-override baseline) x lamda in {4,5,8,10,16}
# -- completes the grid alongside the eps9/10/12 sweep. Waits for that sweep's own
# SWEEP_ALL_DONE marker (single GPU, must run after, not concurrently) before starting.
set -e
cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
DRIVER_LOG=results/CIFAR100/sweep_traineps_lamda_20260722_driver.log

until grep -q "SWEEP_ALL_DONE" "$DRIVER_LOG"; do sleep 30; done

for lamda in 4 5 8 10 16; do
  echo "=== START featdir_teacher300ep_eps8 lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
  $PY -u main.py --config_name featdir_teacher300ep_eps8.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda "$lamda" >> "$DRIVER_LOG" 2>&1
  echo "=== DONE featdir_teacher300ep_eps8 lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
done
echo "ALL_SWEEPS_COMPLETE $(date)" >> "$DRIVER_LOG"
