#!/bin/bash
# RAW-STUDENT gamma sweep (user, 2026-07-09): user suspects the tau_c effect only shows once
# student_norm is OFF (norm student's head learns the OPPOSITE per-class structure, rho=-0.77,
# so it absorbs/cancels injected tau_c; raw student head aligns +0.88 -> tau_c may actually bite).
# Fills the tau16 gamma axis for rawstudent: {0.0, 0.1} already run by run_tauclass_rawstudent.sh;
# here gamma {0.05, 0.25, 0.5} (0.5 = strong dose for dose-response). 10-step, seed 0.
# Chains behind the rawstudent basetau sweep (waits for RAWSTUDENT_TAUSWEEP_DONE).
# Results -> results/CIFAR100/temp_tauclass_fixed_10step_rawstudent/output.log (parse by gamma).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT

until grep -q "RAWSTUDENT_TAUSWEEP_DONE" results/CIFAR100/rawstudent_tauclass_chain.log 2>/dev/null; do
  sleep 120
done

for g in 0.05 0.25 0.5; do
  echo "=== rawstudent tauclass gamma $g $(date) ===" >> results/CIFAR100/rawstudent_tauclass_chain.log
  $PY -u main.py --config_name temp_tauclass_fixed_10step_rawstudent.yaml --dataset CIFAR100 --seed 0 --gamma $g \
    > results/CIFAR100/rawstudent_gamma${g}_driver.log 2>&1
done

echo "RAWSTUDENT_GAMMASWEEP_DONE $(date)" >> results/CIFAR100/rawstudent_tauclass_chain.log
