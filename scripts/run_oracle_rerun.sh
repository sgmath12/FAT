#!/bin/bash
# Oracle cell RERUN (2026-07-14 11:55): first run silently fell through to the random branch
# (whitelist bug, numbers bit-identical to random-k50 61.40/30.06/25.50 -- also an accidental
# exact-reproducibility check). Branch fixed to startswith("pca_"). Behind SPAN_10STEP_DONE.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "SPAN_10STEP_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== ORACLE RERUN pcakdstudent eta=50 START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_span_pcakdstudent.yaml --dataset CIFAR100 --seed 0 --eta 50 \
  > results/CIFAR100/oracle_rerun_driver.log 2>&1
echo "ORACLE_RERUN_DONE $(date)" >> $LOG
