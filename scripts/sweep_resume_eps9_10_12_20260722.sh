#!/bin/bash
# 2026-07-22: resumes the original sweep after eps8 (promoted to run first) finishes.
# eps9 group only had lamda={4,5,8,10} completed before the lamda=16 run was cancelled
# mid-flight -- redo lamda=16, then run eps10 and eps12 groups (untouched, 5 each).
set -e
cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
DRIVER_LOG=results/CIFAR100/sweep_traineps_lamda_20260722_driver.log

until grep -q "EPS8_GROUP_DONE" "$DRIVER_LOG"; do sleep 30; done

echo "=== START featdir_teacher300ep_eps9 lamda=16 $(date) ===" >> "$DRIVER_LOG"
$PY -u main.py --config_name featdir_teacher300ep_eps9.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 16 >> "$DRIVER_LOG" 2>&1
echo "=== DONE featdir_teacher300ep_eps9 lamda=16 $(date) ===" >> "$DRIVER_LOG"

for eps_cfg in featdir_teacher300ep_eps10 featdir_teacher300ep_eps12; do
  for lamda in 4 5 8 10 16; do
    echo "=== START ${eps_cfg} lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
    $PY -u main.py --config_name "${eps_cfg}.yaml" --dataset CIFAR100 --seed 0 --eta 350 --lamda "$lamda" >> "$DRIVER_LOG" 2>&1
    echo "=== DONE ${eps_cfg} lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
  done
done
echo "ALL_SWEEPS_COMPLETE $(date)" >> "$DRIVER_LOG"
