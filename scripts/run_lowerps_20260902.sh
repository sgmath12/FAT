#!/usr/bin/env bash
# Extend the operating curve to eps_tr = 6 and 7 /255 (2026-09-02).  See the config headers.
# CIFAR-10 first: that is the dataset where Generalist++ still has the higher clean accuracy, and the
# extrapolation says eps_tr = 6 passes it.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run () { echo "=== $(date '+%m-%d %H:%M') start $1/$2 ==="
  $PY -u main.py --config_name "$2.yaml" --dataset "$1" --seed 0 > "logs/$1_$2.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1/$2 (exit $?) ==="; }
run CIFAR10  champ_eps6
run CIFAR10  champ_eps7
run CIFAR100 champ_eps6
run CIFAR100 champ_eps7
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
