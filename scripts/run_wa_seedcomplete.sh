#!/bin/bash
# Complete the WA 2x3-seed paired table (2026-07-14 23:00): baseline+WA seeds 1,2 + k350+WA seed2.
# Current: k350+WA 43.79/43.82 (s0/s1, ultra-stable) vs control 43.38 (s0 only, edge +0.4 unpaired).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
for s in 1 2; do
  echo "=== baseline_10step_WA seed $s START $(date) ===" >> $LOG
  $PY -u main.py --config_name temp_baseline_10step_wa.yaml --dataset CIFAR100 --seed $s \
    > results/CIFAR100/baseline_10step_wa_seed${s}_driver.log 2>&1
  echo "=== baseline_10step_WA seed $s DONE $(date) ===" >> $LOG
done
echo "=== k350 WA seed2 START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 2 --eta 350 \
  > results/CIFAR100/k350_wa_seed2_driver.log 2>&1
CK=CIFAR100/checkpoint/featdir_span_random_10step_wa
cp $CK/feat_direction_last.pkl $CK/k350wa_seed2_last.pkl 2>/dev/null
echo "WA_SEEDCOMPLETE_DONE $(date)" >> $LOG
