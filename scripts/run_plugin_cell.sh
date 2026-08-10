#!/bin/bash
# Plug-in cell: baseline KL loss + k350 direction regularizer (behind K350_LAMDA_DONE).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "K350_LAMDA_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== plugin baselineKL+k350dir START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_plugin.yaml --dataset CIFAR100 --seed 0 --eta 350 \
  > results/CIFAR100/featdir_plugin_k350_driver.log 2>&1
echo "PLUGIN_CELL_DONE $(date)" >> $LOG
