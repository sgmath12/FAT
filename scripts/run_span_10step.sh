#!/bin/bash
# 10-step verification of the k-lever (user, 2026-07-14 11:15), behind PCA_SPAN_CELLS_DONE.
# Cells: k350 (3-step best, H 41.42), k100 (plateau center), k350+lamda100 (combo shot).
# Bar = baseline_10step 42.18 (62.66/31.79/cw ~26.95). ~1h/run at 10-step.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "PCA_SPAN_CELLS_DONE" $LOG 2>/dev/null; do sleep 120; done
run () { echo "=== 10step span k=$1 lamda=$2 START $(date) ===" >> $LOG; \
  $PY -u main.py --config_name featdir_span_random_10step.yaml --dataset CIFAR100 --seed 0 --eta $1 --lamda $2 \
    > results/CIFAR100/span10_k$1_l$2_driver.log 2>&1; \
  echo "=== 10step span k=$1 lamda=$2 DONE $(date) ===" >> $LOG; }
run 350 0.0
run 100 0.0
run 350 100.0
echo "SPAN_10STEP_DONE $(date)" >> $LOG
