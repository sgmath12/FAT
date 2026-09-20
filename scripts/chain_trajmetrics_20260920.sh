#!/usr/bin/env bash
# 2026-09-20.  Waits for the seed-1 teacher trajectory, then measures the teacher-only metrics on its six
# snapshots and prints the window the frozen rule selects.  No student is trained here: the window goes
# into notes/teacher_selection.md first.  Seed 2 is deliberately not run yet.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=92544
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') clean_traj_s1 finished ==="
D=CIFAR100/checkpoint/clean_traj_s1_seed1
ls -la $D
$PY scripts/teacher_metrics.py --out writting_docs/paper/notes/teacher_metrics_traj_s1.json \
  --checkpoints $D/clean_ep20.pkl $D/clean_ep40.pkl $D/clean_ep100.pkl $D/clean_ep150.pkl \
                $D/clean_ep200.pkl $D/clean_ep300.pkl
$PY scripts/teacher_window.py --metrics writting_docs/paper/notes/teacher_metrics_traj_s1.json \
  --checkpoints $D/clean_ep20.pkl $D/clean_ep40.pkl $D/clean_ep100.pkl $D/clean_ep150.pkl \
                $D/clean_ep200.pkl $D/clean_ep300.pkl
