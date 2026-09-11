#!/usr/bin/env bash
# Resume only the unfinished tail of chain_restart_20260909 after the prioritized t50 geometry run.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

while pgrep -f '[m]ain.py --config_name margingeom_t50_g10.yaml' >/dev/null; do
  sleep 30
done

for entry in mart_ourrecipe_100ep \
             trades_100ep mart_100ep \
             trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep \
             ship_ep10 ship_ep25 ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 ship_ep200; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$entry seed 0 ==="
  "$PY" -u main.py --config_name "${entry}.yaml" --dataset CIFAR100 --seed 0 \
    > "logs/CIFAR100_${entry}_s0.log" 2>&1
  status=$?
  echo "=== $(date '+%m-%d %H:%M') done $entry seed 0 (exit $status) ==="
done
