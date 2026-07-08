#!/bin/bash
# Seed-verify for the 10-step tauclass_fixed gamma=0.05 spike (user, 2026-07-08).
# seed0 already done (H 42.69). Run seeds 1,2 to test whether +0.51 over baseline_10step (42.18) survives.
# Appends to results/CIFAR100/temp_tauclass_fixed_10step/output.log; parse by (gamma=0.05, seed).
# Paired bar: baseline10step_seed{1,2}_driver.log (already on disk).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
for s in 1 2; do
  echo "=== 10step gamma 0.05 seed $s $(date) ==="
  $PY -u main.py --config_name temp_tauclass_fixed_10step.yaml --dataset CIFAR100 --seed $s --gamma 0.05 \
    > results/CIFAR100/tauclass_g005_seed${s}_driver.log 2>&1
done
echo "TAUCLASS_G005_SEEDVERIFY_DONE $(date)"
