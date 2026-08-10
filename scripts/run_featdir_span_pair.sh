#!/bin/bash
# Exists-and-unique subspace pair (2026-07-13 night), chained behind TEMP_DIRATTACK_DONE.
# 1-ep smoke gates each cell before its full run.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_round2_chain.log
until grep -q "TEMP_DIRATTACK_DONE" $LOG 2>/dev/null; do sleep 120; done
for cfg in featdir_span_teacher featdir_span_random; do
  echo "=== $cfg SMOKE START $(date) ===" >> $LOG
  if $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 --epochs 1 \
      > results/CIFAR100/${cfg}_smoke_driver.log 2>&1; then
    echo "=== $cfg smoke OK, full run START $(date) ===" >> $LOG
    $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 \
      > results/CIFAR100/${cfg}_driver.log 2>&1
    echo "=== $cfg DONE $(date) ===" >> $LOG
  else
    echo "!!! $cfg SMOKE FAILED, skipping $(date)" >> $LOG
  fi
done
echo "FEATDIR_SPAN_PAIR_DONE $(date)" >> $LOG
