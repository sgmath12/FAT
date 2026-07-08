#!/bin/bash
# INPUT-DEPENDENT ideas #2/#3 smoke sweep (queued behind the WA+lamda sweep, waits on its PID).
#   #2 temp_reweight: per-sample KD loss reweight by teacher instability. gamma {1,4} (gamma=0 == baseline 41.77).
#   #3 temp_padapt : learned per-sample normalization strength p(x). 1 run; p-stats logged per epoch.
#   All: tau=16, seed0, isolated (lamda=0, weight_avg=False), 50ep.
#   Results -> results/CIFAR100/temp_reweight/output.log (parse by gamma), results/CIFAR100/temp_padapt/output.log
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

WAIT_PID=${1:-0}
if [ "$WAIT_PID" != "0" ]; then
  echo "waiting for PID $WAIT_PID (wa_lamda sweep) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "sweep finished, starting $(date)"
fi

for g in 1 4; do
  echo ">>> temp_reweight gamma=$g $(date)"
  $PY -u main.py --config_name temp_reweight.yaml --gamma "$g" --tau 16 --seed 0 --dataset CIFAR100 >/dev/null 2>&1
done

echo ">>> temp_padapt $(date)"
$PY -u main.py --config_name temp_padapt.yaml --tau 16 --seed 0 --dataset CIFAR100 >/dev/null 2>&1

echo "############ C100 REWEIGHT+PADAPT DONE $(date) ############"
