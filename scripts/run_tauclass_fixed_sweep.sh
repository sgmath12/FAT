#!/bin/bash
# FIXED per-class tau sweep (2026-07-07, replaces the killed bilevel mobility sweep): no learning,
# tau_c set once from per-class teacher gradnorm (diag_perclass_teacher.npz), geomean pinned at 16.
# Full 50k train (val False) -> fair bar = 50k baseline tau16 seed0 H 41.77.
# gamma > 0: hard/high-gnorm classes softer (tau up to ~20/25); gamma < 0: sharper (down to ~10/7).
# All runs append to results/CIFAR100/temp_tauclass_fixed/output.log; parse by gamma in the
# Experiment Configuration line. ~1h/run, 4 runs.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
for g in 0.25 -0.25 0.5 -0.5; do
  echo "=== run: gamma $g $(date) ==="
  $PY -u main.py --config_name temp_tauclass_fixed.yaml --dataset CIFAR100 --seed 0 --gamma $g \
    > results/CIFAR100/tauclass_fixed_driver.log 2>&1
done
echo "TAUCLASS_FIXED_DONE $(date)"
