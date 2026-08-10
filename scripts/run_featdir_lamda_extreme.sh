#!/bin/bash
# EXTREME lamda probe (user, 2026-07-13 19:15): the consistency term is doubly damped --
# .mean() over N*C divides by 100 classes AND annealing=(ep/50)^2 gates it -- so lamda 3 has
# effective per-sample weight ~0.03. lamda {10, 100} = the properly calibrated dose (100 ~= main
# loss magnitude late in training). Context: lamda sweep {0.3,1,3} gave +0.3~0.5 pgd on featdir
# (best 40.42 @ 0.3), ZERO effect on baseline control (41.75) -> featdir-specific routing lever.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_round2_chain.log
until grep -q "FEATDIR_TAU_DONE" $LOG 2>/dev/null; do sleep 120; done
for l in 10.0 100.0; do
  echo "=== featdir lamda $l START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir.yaml --dataset CIFAR100 --seed 0 --lamda $l \
    > results/CIFAR100/featdir_lamda${l}_driver.log 2>&1
  echo "=== featdir lamda $l DONE $(date) ===" >> $LOG
done
echo "FEATDIR_LAMDA_EXTREME_DONE $(date)" >> $LOG
