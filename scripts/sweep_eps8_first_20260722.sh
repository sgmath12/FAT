#!/bin/bash
# 2026-07-22: eps8 group promoted to run FIRST (user request) -- lamda in {5,8,10,16}.
# lamda=4 SKIPPED: it's a duplicate of the earlier manual champion run
# (featdir_span_random_10step_wa_teacher300ep.yaml, same checkpoint/seed/eta, train_eps
# 8/255 == config.eps default) -- clean=60.13, cw=29.70, H(cw)=39.76, already recorded.
# The original eps9/10/12 sweep was cancelled mid-way (eps9 lamda16 was killed in-flight,
# its partial log removed) and will resume afterward via sweep_resume_eps9_10_12_20260722.sh.
set -e
cd /mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
DRIVER_LOG=results/CIFAR100/sweep_traineps_lamda_20260722_driver.log

for lamda in 5 8 10 16; do
  echo "=== START featdir_teacher300ep_eps8 lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
  $PY -u main.py --config_name featdir_teacher300ep_eps8.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda "$lamda" >> "$DRIVER_LOG" 2>&1
  echo "=== DONE featdir_teacher300ep_eps8 lamda=${lamda} $(date) ===" >> "$DRIVER_LOG"
done
echo "EPS8_GROUP_DONE $(date)" >> "$DRIVER_LOG"
