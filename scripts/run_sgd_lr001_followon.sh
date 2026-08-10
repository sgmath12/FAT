#!/bin/bash
# Follow-on cell added 2026-07-29 21:30: SGD lr0.1 DIVERGED (clean collapsed to 1.6% at the
# OneCycle peak, final 30.99/16.46 vs champion 62.75/33.96). lr0.05 is likely to fail the same
# way, so this appends a genuinely low-LR cell to keep one usable "does SGD work at all" point.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
CHAIN=results/CIFAR100/sgd_lrsweep_20260729.log
until grep -q "SGD_LRSWEEP_DONE" $CHAIN 2>/dev/null; do sleep 60; done
echo "=== sgd lr001 (lr 0.01, eta 350) START $(date) ===" >> $CHAIN
$PY -u main.py --config_name featdir_wa_sgd_lr001.yaml --dataset CIFAR100 --seed 0 \
    --eta 350 --lamda 4.0 > results/CIFAR100/sgd_lr001_driver.log 2>&1
echo "=== sgd lr001 DONE $(date) ===" >> $CHAIN
