#!/usr/bin/env bash
# 2026-09-23.  Two things the last review round asked for, in order of value:
#  1. the radius pair on CIFAR-10 at the final recipe.  The +2.19 clean gain is reproduced over three
#     seeds on CIFAR-100 only, so a second dataset closes "is this one dataset?".
#  2. a second seed for the signal control (same multiset of radii, reassigned by difficulty).  That
#     control is what separates "the signal matters" from "any spread of radii matters", and it rests
#     on one run: 60.90/28.06 against sensitivity's 62.17/28.86 at 100 epochs.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run() { echo "=== $(date '+%m-%d %H:%M') start $2/$1 seed $3 ==="; \
  $PY -u main.py --config_name $1.yaml --dataset $2 --seed $3 > logs/$2_$1_s$3.log 2>&1; \
  echo "=== $(date '+%m-%d %H:%M') done $2/$1 seed $3 (exit $?) ==="; }
run cfa_eps8_150ep_p0 CIFAR10 0
run cfa_eps8_150ep    CIFAR10 0
run champ_diffrank    CIFAR100 1
run ladder_angeps_waawp_100ep CIFAR100 1
