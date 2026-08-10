#!/bin/bash
# k-CURVE (2026-07-14 morning): "how much of the teacher's direction should you follow?"
# featdir_span_random with k = --eta {25, 50, 200, 350}; existing points k=100 (30.12/30.50),
# k=512 = plain featdir (28.91). lamda 0 (clean isolation). Parse featdir_span_random/output.log
# by eta. 3-seed for the combo SKIPPED per user (curve first).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
for k in 25 50 200 350; do
  echo "=== span_random k=$k START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random.yaml --dataset CIFAR100 --seed 0 --eta $k \
    > results/CIFAR100/spank_k${k}_driver.log 2>&1
  echo "=== span_random k=$k DONE $(date) ===" >> $LOG
done
echo "SPAN_K_CURVE_DONE $(date)" >> $LOG
