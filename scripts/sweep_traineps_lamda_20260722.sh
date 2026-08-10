#!/bin/bash
# 2026-07-22: train_eps x lamda sweep on the 300ep-teacher champion base.
# train_eps in {9,10,12}/255, lamda in {4,5,8,10,16} -- 15 runs, sequential (single GPU).
# Each run's log filename is now timestamped (main.py fix, 2026-07-22) so nothing
# overwrites within a shared --config_name; the "Experiment Configuration" line in each
# log records the exact train_eps/lamda used for that run.
set -e
cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
DRIVER_LOG=results/CIFAR100/sweep_traineps_lamda_20260722_driver.log

for eps_cfg in featdir_teacher300ep_eps9 featdir_teacher300ep_eps10 featdir_teacher300ep_eps12; do
  for lamda in 4 5 8 10 16; do
    echo "=== START ${eps_cfg} lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
    $PY -u main.py --config_name "${eps_cfg}.yaml" --dataset CIFAR100 --seed 0 --eta 350 --lamda "$lamda" >> "$DRIVER_LOG" 2>&1
    echo "=== DONE ${eps_cfg} lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
  done
done
echo "SWEEP_ALL_DONE $(date)" >> "$DRIVER_LOG"
