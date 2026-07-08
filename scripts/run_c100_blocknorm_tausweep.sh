#!/bin/bash
# block_norm tau sweep (arch is different -> re-find optimum). seed0. tau16 already running separately.
# near-16 first (12,20) then wider (8,24,4). Results append to blocknorm_temp/output.log (parse by tau).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for tau in 12 20 8 24 4; do
  $PY -u main.py --config_name blocknorm_temp.yaml --tau "$tau" --seed 0 --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ BLOCKNORM tau sweep {12,20,8,24,4} DONE $(date) ############"
