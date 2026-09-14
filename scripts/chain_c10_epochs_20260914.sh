#!/usr/bin/env bash
# 2026-09-14 22:00.  Replaces chain_150ep_20260914.sh (killed while adaad_natinit_stack_100ep ran; that
# python keeps running).  CIFAR-10 CFA vs ARREST full match at 50 and 200 epochs, 8/255; then the rest.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=15341
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') adaad_natinit_stack_100ep finished ==="
for entry in CIFAR10:cfa_eps8_50ep CIFAR10:arrest_full_eps8_50ep CIFAR10:cfa_eps8_200ep CIFAR10:arrest_full_eps8_200ep \
             CIFAR100:adaadigdm_natinit_stack_100ep CIFAR100:consistency_natinit_stack_100ep \
             CIFAR100:lbgat_natinit_stack_100ep \
             CIFAR100:consistency_natinit_100ep CIFAR100:lbgat_natinit_100ep \
             CIFAR100:adr_natinit_200ep CIFAR100:adr_natinit_stack_200ep \
             CIFAR100:consistency_full_100ep CIFAR100:adr_full_100ep; do
  DS="${entry%%:*}"; cfg="${entry##*:}"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
done
