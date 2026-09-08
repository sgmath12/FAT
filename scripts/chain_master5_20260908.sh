#!/usr/bin/env bash
# 2026-09-08 22:30.  Full factorial: 4 targets x {50, 200}-epoch teacher x {no stack, stack} x
# {CIFAR-10, CIFAR-100}, all at 50 student epochs and eps 8/255 so that the only thing separating a
# stacked cell from its twin is weight averaging and the AWP proxy.  Twelve of the 32 cells were
# already measured; these are the remaining twenty, ordered so the frame-deciding block lands first.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1708996
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in c10_ceatk_kdasym_t4 \
           c100_t200_anchor_stack c100_t200_logitmse_stack c100_t200_kdsym_stack c100_t200_kdasym_stack \
           c100_t50_anchor_stack c100_t50_logitmse_stack c100_t50_kdsym_stack c100_t50_kdasym_stack \
           c10_t50_anchor c10_t50_logitmse c10_t50_kdsym c10_t50_kdasym \
           c10_t200_anchor_stack c10_t200_logitmse_stack c10_t200_kdsym_stack c10_t200_kdasym_stack \
           c10_t50_anchor_stack c10_t50_logitmse_stack c10_t50_kdsym_stack c10_t50_kdasym_stack \
           logitmse_100ep kdsym_t16_100ep kdsym_stack_t16 logitmse_stack_100ep \
           adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep \
           trades_100ep mart_100ep trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep \
           clean_5ep clean_10ep clean_20ep clean_40ep \
           ship_ep10 ship_ep25 ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 ship_ep200; do
  DS=CIFAR100; case "$cfg" in c10_*) DS=CIFAR10;; esac
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
