#!/bin/bash
# KL-outer/dir-attack cell, chained behind the extreme-lamda probe (2026-07-13 evening).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_round2_chain.log
until grep -q "FEATDIR_LAMDA_EXTREME_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== temp_dirattack START $(date) ===" >> $LOG
$PY -u main.py --config_name temp_dirattack.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/temp_dirattack_driver.log 2>&1
echo "TEMP_DIRATTACK_DONE $(date)" >> $LOG
