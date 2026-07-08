#!/bin/bash
# UNCONDITIONAL dense decompw gamma sweep (chained behind overnight chain).
# Overnight already runs gamma {1, 4, -1.01}; this fills {0.5, 2, -0.5, -2}
# -> full grid both signs: +{0.5,1,2,4} / -{0.5,1,2}. tau16 seed0 isolated.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID=${1:-0}
if [ "$WAIT_PID" != "0" ]; then
  echo "waiting for PID $WAIT_PID (overnight chain) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 120; done
fi
for g in 0.5 2 -0.5 -2; do
  echo ">>> temp_decompw DENSE gamma=$g $(date)"
  $PY -u main.py --config_name temp_decompw.yaml --gamma "$g" --tau 16 --seed 0 --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ DECOMPW-DENSE DONE $(date) ############"
