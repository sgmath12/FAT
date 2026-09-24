#!/usr/bin/env bash
# 2026-09-24.  Splitting the stack ladder into a 2x2.  tab:ladder adds sensitivity matching, then weight
# averaging, then AWP in one order, so it reports increments and says so, but it never runs AWP without
# WA and therefore cannot separate the two components from their interaction.  The factorial is
# {WA off, on} x {AWP off, on} at p = 1; three of four cells already exist at each setting, and the AWP-
# only cell is the hole.  CIFAR-10 is missing its no-stack cell as well: the 09-22 chain that was meant
# to run it passed --dataset CIFAR100, so only the CIFAR-100 copy was produced.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run() { echo "=== $(date '+%m-%d %H:%M') start $2/$1 ==="; \
  $PY -u main.py --config_name $1.yaml --dataset $2 --seed 0 > logs/$2_$1.log 2>&1; \
  echo "=== $(date '+%m-%d %H:%M') done $2/$1 (exit $?) ==="; }
run cfa_awponly_eps88_100ep CIFAR100   # completes the 8.8/255 factorial, the regime tab:ladder uses
run cfa_awponly_eps8_100ep  CIFAR100
run cfa_nostack_eps8_100ep  CIFAR10
run cfa_awponly_eps8_100ep  CIFAR10
