#!/usr/bin/env bash
# 2026-09-13 20:50.  Replaces chain_after_reboot_20260913.sh (killed while mart_natinit_stack_100ep ran;
# that python keeps running).  ARREST full match at 8/255 on both datasets first, then the rest.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=5858
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') mart_natinit_stack_100ep finished ==="
for entry in CIFAR10:arrest_full_eps8_100ep CIFAR100:arrest_full_eps8_100ep \
             CIFAR100:ard_natinit_stack_100ep \
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
