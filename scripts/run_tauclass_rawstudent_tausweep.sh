#!/bin/bash
# Raw-student BASE-TAU sweep (user, 2026-07-09): tau16 may be miscalibrated for the RAW student
# (basetau sweep 7/8 was normstudent-only), so gamma comparisons at tau16 alone are unfair.
# Waits for the gamma pair (run_tauclass_rawstudent.sh) to finish, then sweeps tau at gamma=0.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT

until grep -q "RAWSTUDENT_TAUCLASS_DONE" results/CIFAR100/rawstudent_tauclass_chain.log 2>/dev/null; do
  sleep 120
done

for t in 8 12 20; do
  echo "=== rawstudent basetau $t gamma 0.0 $(date) ===" >> results/CIFAR100/rawstudent_tauclass_chain.log
  $PY -u main.py --config_name temp_tauclass_fixed_10step_rawstudent.yaml --dataset CIFAR100 --seed 0 --gamma 0.0 --tau $t \
    > results/CIFAR100/rawstudent_basetau_t${t}_driver.log 2>&1
done
echo "RAWSTUDENT_TAUSWEEP_DONE $(date)" >> results/CIFAR100/rawstudent_tauclass_chain.log
