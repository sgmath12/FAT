#!/bin/bash
# 3-seed verify of the TIE candidate (2026-07-14 morning): featdir_span_random + lamda 100
# seed0 = 63.19/30.97 H 41.56 (bar 41.77). Pre-registered rule: paired vs baseline seeds
# (41.77/?/?), tie claim needs mean gap within noise AND no sign surprises.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
for s in 1 2; do
  echo "=== combo span_random+lamda100 seed $s START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random.yaml --dataset CIFAR100 --seed $s --lamda 100.0 \
    > results/CIFAR100/combo_seed${s}_driver.log 2>&1
  echo "=== combo seed $s DONE $(date) ===" >> $LOG
done
echo "COMBO_SEEDVERIFY_DONE $(date)" >> $LOG
