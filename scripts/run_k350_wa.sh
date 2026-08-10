#!/bin/bash
# Robust-leaning variant: k350 10-step + WA (behind PLUGIN_CELL_DONE).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "PLUGIN_CELL_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== 10step k350 WA START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 \
  > results/CIFAR100/k350_wa_driver.log 2>&1
echo "K350_WA_DONE $(date)" >> $LOG
