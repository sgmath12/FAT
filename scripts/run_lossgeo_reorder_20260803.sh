#!/usr/bin/env bash
# Reordered per user: the alpha-1 plain+cosine cell (the one that separates the architecture axis
# from the loss axis) jumps ahead of lossgeo_l2.
#   running : lossgeo_cos     (alpha 0, plain, cosine)
#   next    : lossgeo_cos_a1  (alpha 1, plain, cosine)  <- fills the 2x2's missing cell
#   last    : lossgeo_l2      (alpha 0, plain, L2)
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
WAIT_PID="${1:?need pid of the running lossgeo_cos python}"
echo "=== $(date '+%m-%d %H:%M') waiting on lossgeo_cos pid $WAIT_PID ==="
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
for c in lossgeo_cos_a1 lossgeo_l2; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== reordered queue complete $(date) ==="
