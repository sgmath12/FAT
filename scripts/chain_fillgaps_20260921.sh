#!/usr/bin/env bash
# 2026-09-21.  Filling published-table holes, cheapest first.  No new claims, no seeds:
#  1. tab:stackboth's missing no-stack cells at 100 epochs, both datasets.
#  2. tab:natteacher's CIFAR-10 teacher-initialization rows, which are the C10 counterpart of the
#     CIFAR-100 rows already there.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run() { echo "=== $(date '+%m-%d %H:%M') start $2/$1 ==="; \
  $PY -u main.py --config_name $1.yaml --dataset $2 --seed 0 > logs/$2_$1.log 2>&1; \
  echo "=== $(date '+%m-%d %H:%M') done $2/$1 (exit $?) ==="; }
run cfa_nostack_eps8_100ep CIFAR100
run cfa_nostack_eps8_100ep CIFAR10
run ard_natinit_100ep CIFAR10
run rslad_natinit_100ep CIFAR10
run adaad_natinit_100ep CIFAR10
run adaadigdm_natinit_100ep CIFAR10
