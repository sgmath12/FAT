#!/bin/bash
# NIGHT TAIL (2026-07-14, chained behind NIGHT_SELFMETRIC_DONE): the two remaining hypothesis
# families -- (A) metric-source uniqueness: featdir_teachkl (teacher-metric KL; partner of
# selfkl); (B) attack-strength interaction: featdir_10step + featdir_selfkl_10step (bar 42.18).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "NIGHT_SELFMETRIC_DONE" $LOG 2>/dev/null; do sleep 120; done
for cfg in featdir_teachkl featdir_10step featdir_selfkl_10step; do
  echo "=== $cfg SMOKE START $(date) ===" >> $LOG
  if $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 --epochs 1 \
      > results/CIFAR100/${cfg}_smoke_driver.log 2>&1; then
    echo "=== $cfg smoke OK, full START $(date) ===" >> $LOG
    $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 \
      > results/CIFAR100/${cfg}_driver.log 2>&1
    echo "=== $cfg DONE $(date) ===" >> $LOG
  else
    echo "!!! $cfg SMOKE FAILED, skipped $(date)" >> $LOG
  fi
done
echo "NIGHT_TAIL_DONE $(date)" >> $LOG
