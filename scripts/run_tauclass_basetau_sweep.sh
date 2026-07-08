#!/bin/bash
# base-tau x gamma sweep (user, 2026-07-08): does the per-class gnorm tau signal change with the
# geomean base temperature? 10-step, seed 0. base tau in {8,12,20}; each folder's gamma=0 == that
# tau's own plain baseline (paired control). gamma>0 softens hard/high-gnorm classes.
# Results -> results/CIFAR100/temp_tauclass_fixed_10step_tau{8,12,20}/output.log (parse by gamma).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
for T in 8 12 20; do
  for g in 0.0 0.05 0.1 0.25; do
    echo "=== base tau $T gamma $g $(date) ==="
    $PY -u main.py --config_name temp_tauclass_fixed_10step_tau${T}.yaml --dataset CIFAR100 --seed 0 --gamma $g \
      > results/CIFAR100/basetau_t${T}_g${g}_driver.log 2>&1
  done
done
echo "TAUCLASS_BASETAU_SWEEP_DONE $(date)"
