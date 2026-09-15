#!/usr/bin/env bash
# 2026-09-15 17:45.  Replaces chain_waonly_20260915.sh; adaadigdm_natinit_stack_100ep was killed 23 min in
# and is requeued below.  CFA sample-wise eps + WA only at 50 epochs, 8 and 8.8/255, both datasets.
# CIFAR-100 8.8/255 is config-identical to ladder_angeps_wa_fh_50ep (61.24 / 28.30); rerun as a repeat.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for entry in CIFAR10:cfa_wa_eps8_50ep CIFAR100:cfa_wa_eps8_50ep CIFAR10:cfa_wa_eps88_50ep CIFAR100:cfa_wa_eps88_50ep \
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
