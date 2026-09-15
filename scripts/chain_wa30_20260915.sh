#!/usr/bin/env bash
# 2026-09-15 21:10.  Replaces chain_wa50_20260915.sh (killed while CIFAR-100 cfa_wa_eps88_50ep ran; that
# python keeps running).  CFA sample-wise eps + WA only at 30 epochs, 8 and 8.8/255, both datasets.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=25664
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') CIFAR-100 cfa_wa_eps88_50ep finished ==="
for entry in CIFAR10:cfa_wa_eps8_30ep CIFAR100:cfa_wa_eps8_30ep CIFAR10:cfa_wa_eps88_30ep CIFAR100:cfa_wa_eps88_30ep \
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
