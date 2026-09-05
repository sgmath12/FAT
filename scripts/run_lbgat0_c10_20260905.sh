#!/usr/bin/env bash
# LBGAT0 ON CIFAR-10 (2026-09-05).
#
# The CIFAR-10 cell sat at chance twice and was nearly reported as a failed reproduction.  It is not
# one.  beta = 6 is LBGAT's CIFAR-100 setting; the authors publish CIFAR-10 under LBGAT0 alone and
# their own script passes `--beta 0`.  We were running a configuration they never run on this dataset,
# and methods.py could not have honoured beta = 0 anyway -- `0.0 or 6.0` is 6.0, so LBGAT0 was
# unreachable until that line was fixed today.
#
# CIFAR-100 stays at beta = 6, which is LBGAT6 and is published, so the two datasets run different
# variants on purpose and tab:main names them.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
echo "=== $(date '+%m-%d %H:%M') start CIFAR10/lbgat0_100ep ==="
$PY -u main.py --config_name lbgat0_100ep.yaml --dataset CIFAR10 --seed 0 > logs/CIFAR10_lbgat0_100ep.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
