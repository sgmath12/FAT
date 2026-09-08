#!/usr/bin/env bash
# 2026-09-08 08:45.  The tie between the anchor and the logit MSE is measured at 50 epochs and
# eps 8/255; everything else in the paper trains at 8.8/255 for 100 epochs.  So: the same two
# objectives at that scale WITHOUT the stack first, then the shared-attack control, then the stacked
# pair, then the rest.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1707805
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in c10_kdasym_t4_50ep \
           t50_anchor t50_logitmse t50_kdsym_t16 t50_kdasym_t4 \
           c10_ceatk_anchor c10_ceatk_logitmse c10_ceatk_kdsym_t16 c10_ceatk_kdasym_t4 \
           logitmse_100ep kdsym_t16_100ep \
           kdsym_stack_t16 logitmse_stack_100ep \
           kdsym_t64 logitmse_fh_lr0p042 \
           adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep \
           trades_100ep mart_100ep trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep \
           clean_5ep clean_10ep clean_20ep clean_40ep \
           ship_ep10 ship_ep25 ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 ship_ep200; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  DS=CIFAR100; case "$cfg" in c10_*) DS=CIFAR10;; esac
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
