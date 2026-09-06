#!/usr/bin/env bash
# The stack-matched baseline: AdaAD at its own recipe plus our weight averaging and AWP.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
echo "=== $(date '+%m-%d %H:%M') start CIFAR100/adaad_wa_awp_100ep ==="
$PY -u main.py --config_name adaad_wa_awp_100ep.yaml --dataset CIFAR100 --seed 0 \
  > logs/CIFAR100_adaad_wa_awp_100ep.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
