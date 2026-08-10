#!/bin/bash
# Raw-student tauclass 2x2 completion (user, 2026-07-09 morning).
# {studentRaw} x {gamma 0 = exact baseline, gamma 0.1 = per-class tau_c}, 10-step KL.
# Mechanism prediction: FLAT (head self-calibrates regardless of student_norm).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT

for g in 0.0 0.1; do
  echo "=== rawstudent tauclass gamma $g $(date) ===" >> results/CIFAR100/rawstudent_tauclass_chain.log
  $PY -u main.py --config_name temp_tauclass_fixed_10step_rawstudent.yaml --dataset CIFAR100 --seed 0 --gamma $g \
    > results/CIFAR100/rawstudent_tauclass_g${g}_driver.log 2>&1
done
echo "RAWSTUDENT_TAUCLASS_DONE $(date)" >> results/CIFAR100/rawstudent_tauclass_chain.log
