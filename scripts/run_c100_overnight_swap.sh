#!/bin/bash
# Overnight DENSE swap batch (chained after the running hard-swap tau-density {14,18,20}).
#   A) SOFT-swap 2D grid: tau{16,12,20} x margin(beta){1,2,4}  -> results/CIFAR100/temp_swapsoft/  (9 runs)
#      (does gentle rectify_soft beat hard swap 42.17? peak tau16 first.)
#   B) HARD-swap finer tau: {13,15,17,19} -> results/CIFAR100/temp_studentNorm_teacherRaw_swap/  (4 runs)
#      (combined grid becomes 8,12,13,14,15,16,17,18,19,20,24 - dense around peak 16.)
# All cells logged by tau + beta in each output.log Experiment Configuration line. ~0.5h/run, ~6.5h total.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
# A) soft-swap grid (peak tau16 first)
for tau in 16 12 20; do
  for m in 1 2 4; do
    $PY -u main.py --config_name temp_swapsoft.yaml --tau "$tau" --beta "$m" --dataset CIFAR100 >/dev/null 2>&1
  done
done
echo "############ C100 SOFT-SWAP grid (tau{16,12,20} x margin{1,2,4}) DONE $(date) ############"
# B) hard-swap finer tau
for tau in 13 15 17 19; do
  $PY -u main.py --config_name temp_studentNorm_teacherRaw_swap.yaml --tau "$tau" --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 OVERNIGHT_SWAP ALL DONE $(date) ############"
