#!/usr/bin/env bash
# Reordered again (2026-09-08 08:00).  The symmetric-KD sweep came back stronger than teacher-only
# softening by 2.09 NRR at tau = 16 and was still rising, so before anything else: find where it
# turns over (tau 32, 64) and re-run the full-stack logit comparison with the stronger KD setup,
# since tab:main's logit row and the "the gap widens with the stack" reading both used the weaker one.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1705660
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in kdsym_t32 kdsym_t64 kdsym_stack_t16 \
           logitmse_fh_lr0p007 logitmse_fh_lr0p042 \
           adaadigdm_ourrecipe_100ep trades_ourrecipe_100ep mart_ourrecipe_100ep \
           trades_100ep mart_100ep trades_natinit_100ep mart_natinit_100ep hat_natinit_50ep \
           clean_5ep clean_10ep clean_20ep clean_40ep \
           ship_ep10 ship_ep25 \
           ship_lr0p007 ship_lr0p014 ship_lr0p03 ship_lr0p042 \
           ship_ep200; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
