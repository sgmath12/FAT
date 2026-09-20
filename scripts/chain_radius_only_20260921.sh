#!/usr/bin/env bash
# 2026-09-20 23:40.  Replaces chain_radius_final_20260920.sh: the seed-1 repeats are dropped.  What is
# left is the one cell the radius claim is missing -- the uniform-radius twin of the 150-epoch 8/255
# result -- and then the two constructed teachers, which are cheap and only feed the matched-accuracy
# comparison.  No further seed work.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=96818
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') cfa_eps8_150ep_p0 finished ==="
for cfg in clean_t50plus50_lowlr clean_t50plus150_lowlr; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
$PY scripts/teacher_metrics.py --out writting_docs/paper/notes/teacher_metrics_constructed.json \
  --checkpoints clean_t50plus50_lowlr clean_t50plus150_lowlr
