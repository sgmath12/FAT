#!/usr/bin/env bash
# 2026-09-20.  "Construct a teacher rather than select one": continue the 50-epoch teacher at a small
# constant learning rate, then read its margin and feature sensitivity.  A student is trained only if a
# constructed teacher lands above the window on BOTH quantities, which is the point of the exercise.
# Queued behind the teacher-selection students and the final-recipe radius pair.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
while pgrep -f 'chain_radius_final_20260920' > /dev/null; do sleep 120; done
echo "=== $(date '+%m-%d %H:%M') radius chain finished ==="
for cfg in clean_t50plus50_lowlr clean_t50plus150_lowlr; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
$PY scripts/teacher_metrics.py --out writting_docs/paper/notes/teacher_metrics_constructed.json \
  --checkpoints clean_t50plus50_lowlr clean_t50plus150_lowlr
echo "=== window rule applied to the seed-0 ladder plus the constructed teachers ==="
$PY scripts/teacher_window.py --metrics writting_docs/paper/notes/teacher_metrics.json \
  --checkpoints clean_20ep clean_40ep clean clean_100ep clean_150ep clean_200ep clean_300ep
