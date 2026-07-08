#!/bin/bash
# Cosine-head student cell (user confirmed design 2026-07-07 23:2x: student = s*cos head, teacher
# unchanged raw/tau16). Chains behind run_tauclass_fixed_round2.sh. Single 3-step 50k run, seed 0.
# Fair bar: 50k baseline tau16 seed0 H 41.77 (63.04/31.23). Watch cos_s per epoch (smoke: 1.85->1.08
# DOWN in 1 ep, vs plain student growing ||W|| 5x over 50 ep).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
while pgrep -f "run_tauclass_fixed_round2.sh" > /dev/null || pgrep -f "main.py --config_name temp_tauclass_fixed" > /dev/null; do sleep 60; done
echo "round2 done, starting coshead $(date)"
$PY -u main.py --config_name temp_coshead.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/coshead_driver.log 2>&1
echo "COSHEAD_DONE $(date)"
