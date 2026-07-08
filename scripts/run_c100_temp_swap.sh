#!/bin/bash
# SWAP-BASELINE sweep: global temp + teacher swap-rectification. student-norm, teacher raw.
#   tau = global temperature. tau16 first (matched vs non-swap baseline 41.62), then density.
#   Results -> results/CIFAR100/temp_studentNorm_teacherRaw_swap/output.log (parse by tau). steps=3.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for tau in 16 12 24 8; do
  $PY -u main.py --config_name temp_studentNorm_teacherRaw_swap.yaml --tau "$tau" --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 TEMP_SWAP baseline sweep DONE $(date) ############"
