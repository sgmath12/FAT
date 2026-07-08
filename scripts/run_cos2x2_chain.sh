#!/bin/bash
# cos 2x2 completion (user, 2026-07-07 23:2x): after round2 + coshead(tau16) finish, run
#   (1) coshead tau sweep {8, 12, 20, 24} (tau16 already chained by run_coshead_chain.sh)
#   (2) costeacher tau16 (norm student + cos teacher)  (3) cosboth tau16 (full directional)
# Fair bar: 50k baseline tau16 H 41.77. All 3-step, 50k, seed 0. ~7 x 27min.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
while pgrep -f "run_tauclass_fixed_round2.sh|run_coshead_chain.sh" > /dev/null || pgrep -f "main.py --config_name temp" > /dev/null; do sleep 60; done
echo "queue clear, starting cos2x2 $(date)"
for t in 8 12 20 24; do
  echo "=== coshead tau $t $(date) ==="
  $PY -u main.py --config_name temp_coshead.yaml --dataset CIFAR100 --seed 0 --tau $t \
    > results/CIFAR100/cos2x2_driver.log 2>&1
done
echo "=== costeacher tau 16 $(date) ==="
$PY -u main.py --config_name temp_costeacher.yaml --dataset CIFAR100 --seed 0 > results/CIFAR100/cos2x2_driver.log 2>&1
echo "=== cosboth tau 16 $(date) ==="
$PY -u main.py --config_name temp_cosboth.yaml --dataset CIFAR100 --seed 0 > results/CIFAR100/cos2x2_driver.log 2>&1
echo "COS2X2_DONE $(date)"
