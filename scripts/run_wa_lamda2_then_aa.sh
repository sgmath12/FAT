#!/bin/bash
# Rebuilt chain (2026-07-15): lamda30 (already running) -> lamda100 -> lamda10 (user add) -> AA.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
while ps aux | grep -q "[m]ain.py"; do sleep 60; done
echo "=== k350wa lamda 30.0 DONE(orphan) $(date) ===" >> $LOG
for l in 100.0 10.0; do
  echo "=== k350wa lamda $l START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda $l \
    > results/CIFAR100/k350wa_lamda${l}_driver.log 2>&1
  echo "=== k350wa lamda $l DONE $(date) ===" >> $LOG
done
echo "=== AA eval START $(date) ===" >> $LOG
$PY -u scripts/eval_aa_ckpts.py > results/CIFAR100/aa_eval_20260715.log 2>&1
echo "WA_LAMDA_AA_DONE $(date)" >> $LOG
