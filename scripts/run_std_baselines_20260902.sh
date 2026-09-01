#!/usr/bin/env bash
# THE THREE STANDARD BASELINES, MEASURED HERE (2026-09-02).
#
# The main tables report only numbers produced in this framework, so PGD-AT, TRADES and MART could not
# be quoted from the literature without reintroducing the cross-codebase mixing those tables exist to
# avoid.  All three share the protocol of the distillation baselines already run -- SGD 0.1, momentum
# 0.9, step decay x0.1 at epochs 70 and 90, 100 epochs, eps 8/255, 10-step attack, random
# initialization, no weight averaging, no AWP -- so the whole baseline block is internally comparable.
#
# TRADES beta = 6, MART lambda = 5, both as published.  Neither uses a teacher.
# CIFAR-100 first: it is the dataset the rest of the paper is built on.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
run () {
  echo "=== $(date '+%m-%d %H:%M') start $1/$2 ==="
  $PY -u main.py --config_name "$2.yaml" --dataset "$1" --seed 0 > "logs/$1_$2.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1/$2 (exit $?) ==="
}
run CIFAR100 pgdat_100ep
run CIFAR100 trades_100ep
run CIFAR100 mart_100ep
run CIFAR10  pgdat_100ep
run CIFAR10  trades_100ep
run CIFAR10  mart_100ep
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
