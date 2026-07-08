#!/bin/bash
# Round 2 (user, 2026-07-07 23:1x): (a) 3-step gap cells gamma {0.05, 0.75} (0.1/0.25 already done);
# (b) 10-step (heavy AT) version at gamma {0.05, 0.1, 0.25, 0.75} -- does heavier AT change the
# per-class-tau verdict? Fair bars: 3-step 41.77, 10-step baseline_10step H 42.18.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
for g in 0.05 0.75; do
  echo "=== 3step gamma $g $(date) ==="
  $PY -u main.py --config_name temp_tauclass_fixed.yaml --dataset CIFAR100 --seed 0 --gamma $g \
    > results/CIFAR100/tauclass_fixed_driver.log 2>&1
done
for g in 0.05 0.1 0.25 0.75; do
  echo "=== 10step gamma $g $(date) ==="
  $PY -u main.py --config_name temp_tauclass_fixed_10step.yaml --dataset CIFAR100 --seed 0 --gamma $g \
    > results/CIFAR100/tauclass_fixed_driver.log 2>&1
done
echo "TAUCLASS_FIXED_ROUND2_DONE $(date)"
