#!/bin/bash
# tau16 LOW-gamma cells (user, 2026-07-08 night): dose-response below the g0.05 signal.
# gamma {0.01, 0.02, 0.035}, 10-step seed 0. Paired bar = baseline10step seed0 H(pgd) 42.18.
# Chains behind the treasure/placebo/g0.1 queue (waits for TREASURE_TAUCLASS_QUEUE_DONE).
# Results append to results/CIFAR100/temp_tauclass_fixed_10step/output.log (parse by gamma).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT

until grep -q "TREASURE_TAUCLASS_QUEUE_DONE" results/CIFAR100/treasure_queue_chain.log 2>/dev/null; do
  sleep 120
done

for g in 0.01 0.02 0.035; do
  echo "=== tau16 low gamma $g $(date) ==="
  $PY -u main.py --config_name temp_tauclass_fixed_10step.yaml --dataset CIFAR100 --seed 0 --gamma $g \
    > results/CIFAR100/tauclass_lowg${g}_driver.log 2>&1
done

echo "TAUCLASS_LOWGAMMA_TAU16_DONE $(date)"
