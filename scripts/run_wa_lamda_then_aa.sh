#!/bin/bash
# GPU-idle filler (user, 2026-07-15 morning): (1) k350+WA+lamda {30,100} 10-step — the lamda x WA
# interaction was never tested (lamda died 3x at k350 but always WITHOUT WA); low expectation
# (lamda historically costs cw) but cheap. (2) Then AA on 4 archived ckpts (the cw-deficit arbiter).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
for l in 30.0 100.0; do
  echo "=== k350wa lamda $l START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda $l \
    > results/CIFAR100/k350wa_lamda${l}_driver.log 2>&1
  echo "=== k350wa lamda $l DONE $(date) ===" >> $LOG
done
echo "=== AA eval START $(date) ===" >> $LOG
$PY -u scripts/eval_aa_ckpts.py > results/CIFAR100/aa_eval_20260715.log 2>&1
echo "WA_LAMDA_AA_DONE $(date)" >> $LOG
