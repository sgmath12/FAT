#!/bin/bash
# GLOBAL learned p ablation (chained behind run_c100_reweight_padapt.sh via PID wait).
#   temp_pglobal: student = ResNet18_zpg, p = sigmoid(scalar), zero-init. tau16 seed0 isolated 50ep.
#   3-way ablation: p=1 fixed (41.77) vs p* global (THIS) vs p(x) per-sample (temp_padapt).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID=${1:-0}
if [ "$WAIT_PID" != "0" ]; then
  echo "waiting for PID $WAIT_PID (reweight+padapt queue) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
echo ">>> temp_pglobal $(date)"
$PY -u main.py --config_name temp_pglobal.yaml --tau 16 --seed 0 --dataset CIFAR100 >/dev/null 2>&1
echo "############ C100 PGLOBAL DONE $(date) ############"
