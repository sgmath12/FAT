#!/usr/bin/env bash
# L1 vs L2 GRADIENT NORM, ON THE SHIPPED RECIPE (2026-09-04).
#
# One cell.  The appendix's answer to "you derive L1 and ship L2" is champ_angeps_gnorm1, which
# carries student_norm True + freeze_lr_epoch 0.65 -- the pre-2026-08-31 directional design, the same
# stale-regime problem that sent the per-sample-eps controls back through the card this morning.
# champ_gnorm1_l2 is the shipped config with featdir_angeps_gnorm = 1 and nothing else changed.
#
# Launched as its own driver rather than inserted into chain_after_controls_20260904, which was
# already running and cannot be edited in place.  main.py holds a per-GPU flock, so this waits;
# flock wake order is not FIFO, so it may land before or after the first cell of the resumed master
# queue.  Either is fine -- nothing downstream depends on the order.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
echo "=== $(date '+%m-%d %H:%M') start CIFAR100/champ_gnorm1_l2 ==="
$PY -u main.py --config_name champ_gnorm1_l2.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_champ_gnorm1_l2.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done CIFAR100/champ_gnorm1_l2 (exit $?) ==="
