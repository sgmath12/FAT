#!/bin/bash
# Chained behind the beta sweep (polls for FEATDIR_BETA_SWEEP_DONE): featdir_gainhead with
# log_g exempted from AdamW wd -- completes the no-decay discriminating test in the featdir
# pipeline (user, 2026-07-13). Own folder featdir_gainhead_nodecay.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_beta_chain.log
until grep -q "FEATDIR_BETA_SWEEP_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== featdir_gainhead_nodecay START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_gainhead_nodecay.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/featdir_gainhead_nodecay_driver.log 2>&1
echo "FEATDIR_GAINHEAD_NODECAY_DONE $(date)" >> $LOG
