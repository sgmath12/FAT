#!/usr/bin/env bash
# 2026-09-20.  Step 2 of the teacher-selection study: two new natural-teacher trajectories with
# epoch-indexed snapshots (20, 40, 100, 150, 200, 300).  Students are NOT queued here on purpose: the
# candidate window has to be marked from teacher metrics alone, and written down, before any student on
# these trajectories is trained.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for cfg in clean_traj_s1 clean_traj_s2; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed "${cfg##*_s}" > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
