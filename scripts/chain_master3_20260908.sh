#!/usr/bin/env bash
# 2026-09-08 08:30.  Order:
#   1. the shared-attack control -- four objectives, one true-label CE-PGD, base regime.  Isolates
#      what is supervised from what the attack ascends, which change together in every cell so far.
#   2. the two full-stack logit cells.  The base regime has logit MSE tied with the anchor inside the
#      seed spread, so whether that tie survives the stack is what decides how the paper frames the
#      layer question.
#   3. the rest of the temperature curve, then the table-5 remainder, then sensitivity and teachers.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1705920
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in ceatk_logitmse ceatk_kdsym_t16 ceatk_kdasym_t4 ceatk_anchor \
           kdsym_stack_t16 logitmse_stack_100ep \
           kdsym_t64 logitmse_fh_lr0p042 \
           adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep \
           trades_100ep mart_100ep trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep \
           clean_5ep clean_10ep clean_20ep clean_40ep \
           ship_ep10 ship_ep25 ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 ship_ep200; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
