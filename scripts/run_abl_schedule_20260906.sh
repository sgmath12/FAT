#!/usr/bin/env bash
# DEFENSIVE ABLATION: is the gain the warm start and the schedule? (2026-09-06)
#
# The warm-start half is already answered -- abl_scratch_init (random init, shipped recipe) lands
# 60.81 / 25.21 against the warm-started 61.29 / 25.36.  These two cells answer the schedule half by
# completing the 2x2 at the base regime, running the anchor and label cross-entropy under PGD-AT's
# own optimizer and schedule (SGD 0.1, piecewise x0.1 at 70 and 90) instead of ours.
#
# Sequential on one GPU, CE first: if our schedule was handicapping CE, that cell is the one that
# changes what the paper has to say, so it should land first.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for cfg in abl_sched_ce_sgdpw_100ep abl_sched_anchor_sgdpw_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
