#!/bin/bash
# Overnight batch (chained AFTER the running smooth_temp T-sweep):
#   1. smooth+swap (golden-logit combo) T{16,12,24,8}, K8 -> results/CIFAR100/smooth_temp_swap/
#   2. swap-baseline finer tau{14,18,20} -> results/CIFAR100/temp_studentNorm_teacherRaw_swap/ (append)
# All params (temperature, smooth_k, tau...) logged in each output.log's Experiment Configuration line.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
# 1) golden-logit combo (eps-ball smooth + swap + temp)
for T in 16 12 24 8; do
  $PY -u main.py --config_name smooth_temp_swap.yaml --temperature "$T" --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 SMOOTH_TEMP_SWAP (K8, T sweep) DONE $(date) ############"
# 2) swap-baseline finer tau density (cheap tail)
for tau in 14 18 20; do
  $PY -u main.py --config_name temp_studentNorm_teacherRaw_swap.yaml --tau "$tau" --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 OVERNIGHT_GOLDEN ALL DONE $(date) ############"
