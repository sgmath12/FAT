#!/usr/bin/env bash
# 2026-09-14.  Replaces chain_75ep_20260914.sh (cancelled before any 75-epoch cell started).
# CFA vs ARREST full match at 150 epochs, training radius 8/255, both datasets; then the rest.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=8807
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') ard_natinit_stack_100ep finished ==="
for entry in CIFAR100:cfa_eps8_150ep CIFAR100:arrest_full_eps8_150ep CIFAR10:cfa_eps8_150ep CIFAR10:arrest_full_eps8_150ep \
             CIFAR100:rslad_natinit_stack_100ep CIFAR100:adaad_natinit_stack_100ep \
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
