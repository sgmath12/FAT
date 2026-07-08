#!/bin/bash
# OVERNIGHT 2026-07-03 (chained behind pglobal, PID passed as $1). ~11 runs x ~0.5h.
#   1. decompw gamma {1,4,-1}: idea A rotation-share weight (+=downweight rot-dominant, -1=upweight)
#   2. reweight gamma {-0.5,-2}: UPWEIGHT teacher-unstable (downweight hurt: 40.58/38.41 < 41.77)
#   3. blocknorm tau24 seeds 1,2 (seed0 was 41.90/cw 26.96 best-cw; need seed mean)
#   4. wa & swap_wa lamda2 seeds 1,2 (submission-pipeline seed check; lamda2 was best 42.99)
# NOTE gamma=-1 exactly is the CLI sentinel and CANNOT be passed -> use -1.01 for "gamma -1".
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

WAIT_PID=${1:-0}
if [ "$WAIT_PID" != "0" ]; then
  echo "waiting for PID $WAIT_PID (pglobal) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
echo "start $(date)"

for g in 1 4 -1.01; do
  echo ">>> temp_decompw gamma=$g $(date)"
  $PY -u main.py --config_name temp_decompw.yaml --gamma "$g" --tau 16 --seed 0 --dataset CIFAR100 >/dev/null 2>&1
done

for g in -0.3 -0.8; do
  echo ">>> temp_reweight gamma=$g (UPWEIGHT) $(date)"
  $PY -u main.py --config_name temp_reweight.yaml --gamma "$g" --tau 16 --seed 0 --dataset CIFAR100 >/dev/null 2>&1
done

for s in 1 2; do
  echo ">>> blocknorm_temp tau=24 seed=$s $(date)"
  $PY -u main.py --config_name blocknorm_temp.yaml --tau 24 --seed "$s" --dataset CIFAR100 >/dev/null 2>&1
done

for s in 1 2; do
  echo ">>> temp_studentNorm_teacherRaw_wa lamda=2 seed=$s $(date)"
  $PY -u main.py --config_name temp_studentNorm_teacherRaw_wa.yaml --tau 16 --lamda 2 --seed "$s" --dataset CIFAR100 >/dev/null 2>&1
done
for s in 1 2; do
  echo ">>> temp_studentNorm_teacherRaw_swap_wa lamda=2 seed=$s $(date)"
  $PY -u main.py --config_name temp_studentNorm_teacherRaw_swap_wa.yaml --tau 16 --lamda 2 --seed "$s" --dataset CIFAR100 >/dev/null 2>&1
done

echo "############ OVERNIGHT 0703 DONE $(date) ############"
