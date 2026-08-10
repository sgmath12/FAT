#!/bin/bash
# Matched-adversary lamda cells behind K350WA_LOWDOSE_DONE: smoke, then lamda {1, 4} with the
# dir+cons hybrid attack (2026-07-15).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "K350WA_LOWDOSE_DONE" $LOG 2>/dev/null; do sleep 120; done
echo "=== dircons SMOKE START $(date) ===" >> $LOG
if $PY -u main.py --config_name k350wa_dircons.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --epochs 1 \
    > results/CIFAR100/dircons_smoke_driver.log 2>&1; then
  echo "=== dircons smoke OK $(date) ===" >> $LOG
  for l in 1.0 4.0; do
    echo "=== dircons lamda $l START $(date) ===" >> $LOG
    $PY -u main.py --config_name k350wa_dircons.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda $l \
      > results/CIFAR100/dircons_lamda${l}_driver.log 2>&1
    echo "=== dircons lamda $l DONE $(date) ===" >> $LOG
  done
else
  echo "!!! dircons SMOKE FAILED $(date)" >> $LOG
fi
echo "DIRCONS_DONE $(date)" >> $LOG
