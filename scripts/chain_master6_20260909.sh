#!/usr/bin/env bash
# 2026-09-09.  The factorial's stack half is dropped: the analysis reads as a ladder of what is
# transferred (soft labels -> shared-temperature KD -> logit MSE -> features), and that ladder does
# not need to be repeated under weight averaging and AWP to make its point.  What is kept from the
# stack side is the pair at the SHIPPED recipe, because tab:main's logit row currently uses the
# weaker teacher-only KD and has to be restated against conventional KD and the logit MSE.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1708996
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in c10_t50_anchor c10_t50_logitmse c10_t50_kdsym c10_t50_kdasym \
           kdsym_stack_t16 logitmse_stack_100ep \
           logitmse_100ep kdsym_t16_100ep \
           clean_5ep clean_10ep clean_20ep clean_40ep \
           adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep \
           trades_100ep mart_100ep trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep \
           ship_ep10 ship_ep25 ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 ship_ep200; do
  DS=CIFAR100; case "$cfg" in c10_*) DS=CIFAR10;; esac
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
