#!/usr/bin/env bash
# CIFAR10: train the clean_200ep teacher, then the champion recipe on top of it.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
echo "=== $(date '+%m-%d %H:%M') start CIFAR10 clean_200ep (teacher) ==="
$PY main.py --config_name clean_200ep.yaml --dataset CIFAR10 --seed 0 > logs/c10_clean_200ep.log 2>&1
rc=$?; echo "=== $(date '+%m-%d %H:%M') teacher done (exit $rc) ==="
if [ $rc -ne 0 ]; then echo "teacher FAILED, aborting"; exit 1; fi
if [ ! -f CIFAR10/checkpoint/clean_200ep/clean_last.pkl ]; then
  echo "teacher checkpoint MISSING at CIFAR10/checkpoint/clean_200ep/clean_last.pkl, aborting"; exit 1; fi
echo "=== $(date '+%m-%d %H:%M') start CIFAR10 champion ==="
$PY main.py --config_name featdir_champ200_100ep.yaml --dataset CIFAR10 --seed 0 > logs/c10_champ.log 2>&1
echo "=== $(date '+%m-%d %H:%M') champion done (exit $?) ==="
