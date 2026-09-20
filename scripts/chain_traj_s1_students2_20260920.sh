#!/usr/bin/env bash
# 2026-09-20 15:05.  Replaces chain_traj_s1_students_20260920.sh, which waited on the teacher main.py
# process and so could have started in the gap between two teacher runs -- before the metrics and the
# window were printed.  This waits on the teacher CHAIN's pid instead, whose last two commands are the
# metric measurement and the window print, so no student can start until the window is on record.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=93846
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') seed-1 teacher ladder and its window are done ==="
grep -A9 'checkpoint *margin' logs/chain_traj_s1_runs_20260920.log || true
for e in 150 200 100 300 40 20; do   # window first, then its neighbours, then the ends
  cfg=tladder_s1_${e}ep
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
