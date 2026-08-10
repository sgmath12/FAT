#!/bin/bash
# AWP cells (2026-07-15 night): (1)(2) isolation pair lamda0, (3)(4) combo with each track's
# best lamda. gamma=5e-3, warmup=5. Bars: baseline+WA 61.77/33.43/cw28.18; k350+WA 62.61/33.67/
# cw28.00; k350+WA+lamda4 62.75/33.96/cw28.41 (current project best); baseline+WA+lamda10
# 61.52/33.79/cw28.46.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log

echo "=== baseline+WA+AWP (isolation) START $(date) ===" >> $LOG
$PY -u main.py --config_name temp_baseline_10step_wa_awp.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/basewa_awp_driver.log 2>&1
echo "=== baseline+WA+AWP (isolation) DONE $(date) ===" >> $LOG

echo "=== k350+WA+AWP (isolation) START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_k350wa_awp.yaml --dataset CIFAR100 --seed 0 --eta 350 \
  > results/CIFAR100/k350wa_awp_driver.log 2>&1
echo "=== k350+WA+AWP (isolation) DONE $(date) ===" >> $LOG

echo "=== k350+WA+AWP+lamda4 (combo) START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_k350wa_awp.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
  > results/CIFAR100/k350wa_awp_lamda4_driver.log 2>&1
echo "=== k350+WA+AWP+lamda4 (combo) DONE $(date) ===" >> $LOG

echo "=== baseline+WA+AWP+lamda10 (combo control) START $(date) ===" >> $LOG
$PY -u main.py --config_name temp_baseline_10step_wa_awp.yaml --dataset CIFAR100 --seed 0 --lamda 10.0 \
  > results/CIFAR100/basewa_awp_lamda10_driver.log 2>&1
echo "=== baseline+WA+AWP+lamda10 (combo control) DONE $(date) ===" >> $LOG

echo "AWP_CELLS_DONE $(date)" >> $LOG
