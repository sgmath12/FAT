#!/bin/bash
# Chained behind the deltanet pilot (2026-07-06): waits for the running temp_deltanet_bilevel
# process to exit, then runs the 45000-train baseline control (see temp_baseline_val45k.yaml --
# the fair comparison bar for ALL post-valfix bilevel runs).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
while pgrep -f "temp_deltanet_bilevel.yaml" > /dev/null; do sleep 60; done
echo "deltanet pilot done, starting baseline_val45k $(date)"
$PY main.py --config_name temp_baseline_val45k.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/baseline_val45k_driver.log 2>&1
echo "BASELINE_VAL45K_DONE $(date)"
