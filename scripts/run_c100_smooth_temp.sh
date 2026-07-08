#!/bin/bash
# eps-ball smoothed teacher + temperature (golden-logit). K=8 (config), sweep T. T16 first (key point).
#   Compare vs plain temp 41.62 / swap-baseline 42.17. ~2x slower (K teacher forwards). steps=3.
#   Results -> results/CIFAR100/smooth_temp/output.log (parse by temperature).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for T in 16 12 24 8; do
  $PY -u main.py --config_name smooth_temp.yaml --temperature "$T" --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 SMOOTH_TEMP (K8, T sweep) DONE $(date) ############"
