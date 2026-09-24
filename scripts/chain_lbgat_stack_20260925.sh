#!/usr/bin/env bash
# 2026-09-25.  tab:baselinestack's LBGAT row: the "own recipe" column is now measured under the
# corrected lr schedule (57.29 / 26.22), but the "+ teacher init + WA + AWP" column beside it
# (53.91 / 25.16) still comes from the shifted one, so the two halves of that row are one schedule
# revision apart.  One 2-hour cell fixes it.  main.py's flock on /tmp/fat_gpu_0.lock keeps this from
# racing the chains already queued.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cfg=lbgat_natinit_stack_100ep
echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
$PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}_lrfix.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
