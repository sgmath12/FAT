#!/bin/bash
# AA on k350+WA+lamda4 winner, queued immediately after the currently running GPU job frees up.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
while ps aux | grep -q "[m]ain.py"; do sleep 60; done
echo "=== AA lamda4 START $(date) ===" >> $LOG
$PY -u scripts/eval_aa_lamda4.py > results/CIFAR100/aa_lamda4.log 2>&1
echo "=== AA lamda4 DONE $(date) ===" >> $LOG
