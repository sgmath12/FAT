#!/bin/bash
# tauclass MOBILITY sweep (2026-07-07): the pilot (tau_meta_lr=100, window 0-10) froze tau_c at
# std 0.031 (0.2% of tau=16, clamp [8,32] untouched) while std was still ACCELERATING and tauc_min
# was falling MONOTONICALLY -- the meta signal is same-signed but the lr*window budget cut it off.
# SGD => std scales ~linearly with lr, so: lr {1000, 3000, 10000} x window 0-10, plus one longer
# window cell (0-25, lr 3000) to test compounding (taunet history: long windows hurt the BACKBONE
# near the OneCycle peak -- watch clean acc on that cell).
# All runs append to results/CIFAR100/temp_tauclass_bilevel/output.log; cells self-label via the
# Experiment Configuration line (tau_meta_lr / bilevel_start / bilevel_end now logged).
# Waits for the currently-running pilot to finish first.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
while pgrep -f "main.py --config_name temp_tauclass_bilevel.yaml" > /dev/null; do sleep 60; done
echo "pilot done, starting mobility sweep $(date)"
for args in "--tau_meta_lr 1000" "--tau_meta_lr 3000" "--tau_meta_lr 10000" "--tau_meta_lr 3000 --bilevel_end 25"; do
  echo "=== run: $args $(date) ==="
  $PY -u main.py --config_name temp_tauclass_bilevel.yaml --dataset CIFAR100 --seed 0 $args \
    > results/CIFAR100/tauclass_mobility_driver.log 2>&1
done
echo "TAUCLASS_MOBILITY_DONE $(date)"
