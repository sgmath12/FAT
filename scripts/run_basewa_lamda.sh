#!/bin/bash
# DOUBLE-CHECK (user, 2026-07-15 13:45): baseline+WA + lamda {1,10,100} 10-step. User recalls
# lamda raising robustness in the ORIGINAL recipe -- which was WA+lamda together; our lamda
# kills were mostly no-WA or featdir. C-damping note: consistency .mean() divides by C=100,
# so lamda {1,10,100} here = old CIFAR-10 scale {0.1,1,10}. Bar: baseline+WA 3-seed
# 61.70/33.43/cw28.42 H 43.37 (s0 61.77/33.43/28.18).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
for l in 1.0 10.0 100.0; do
  echo "=== basewa lamda $l START $(date) ===" >> $LOG
  $PY -u main.py --config_name temp_baseline_10step_wa.yaml --dataset CIFAR100 --seed 0 --lamda $l \
    > results/CIFAR100/basewa_lamda${l}_driver.log 2>&1
  echo "=== basewa lamda $l DONE $(date) ===" >> $LOG
done
echo "BASEWA_LAMDA_DONE $(date)" >> $LOG
