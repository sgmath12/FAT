#!/bin/bash
# Baseline+WA 10-step control (behind NIGHT3_DONE) + k350+WA seed 1 (verify the breakthrough).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "NIGHT3_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== baseline_10step_WA control START $(date) ===" >> $LOG
$PY -u main.py --config_name temp_baseline_10step_wa.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/baseline_10step_wa_driver.log 2>&1
echo "=== baseline_10step_WA control DONE $(date) ===" >> $LOG
echo "=== k350 WA seed1 START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 1 --eta 350 \
  > results/CIFAR100/k350_wa_seed1_driver.log 2>&1
CK=CIFAR100/checkpoint/featdir_span_random_10step_wa
cp $CK/feat_direction_last.pkl $CK/k350wa_seed1_last.pkl 2>/dev/null
echo "WA_CONTROL_DONE $(date)" >> $LOG
