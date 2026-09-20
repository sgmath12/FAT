#!/usr/bin/env bash
# 2026-09-20.  The seed-1 teacher ladder, one converged run per epoch count, which is how the seed-0
# ladder of tab:teacherladder_values was built.  Replaces the save_epochs snapshot attempt: under a
# one-cycle schedule a mid-run snapshot is a model at the schedule's high-lr midpoint, measured at
# 59-63% clean accuracy on the way to 78%, so those snapshots are not N-epoch teachers.
# 810 epochs in total, about 1.6 h.  Students wait until the window is written down.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for e in 20 40 100 150 200 300; do
  cfg=clean_s1_${e}ep
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 1 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
D=CIFAR100/checkpoint
$PY scripts/teacher_metrics.py --out writting_docs/paper/notes/teacher_metrics_s1.json \
  --checkpoints clean_s1_20ep_seed1 clean_s1_40ep_seed1 clean_s1_100ep_seed1 \
                clean_s1_150ep_seed1 clean_s1_200ep_seed1 clean_s1_300ep_seed1
$PY scripts/teacher_window.py --metrics writting_docs/paper/notes/teacher_metrics_s1.json \
  --checkpoints clean_s1_20ep_seed1 clean_s1_40ep_seed1 clean_s1_100ep_seed1 \
                clean_s1_150ep_seed1 clean_s1_200ep_seed1 clean_s1_300ep_seed1
