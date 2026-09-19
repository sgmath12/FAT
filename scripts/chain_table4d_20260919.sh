#!/usr/bin/env bash
# 2026-09-19 10:05.  Replaces chain_table4c_20260919.sh, adding the cosine-teacher ladder cell second:
# it decides between the two selection rules of notes/teacher_selection.md, and costs one hour.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=76590
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') adaadigdm_natinit_stack_100ep finished ==="
for cfg in tladder_clean_cos200ep consistency_natinit_stack_100ep lbgat_natinit_stack_100ep \
           adr_natinit_stack_200ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
