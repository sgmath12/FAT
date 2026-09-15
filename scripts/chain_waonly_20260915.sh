#!/usr/bin/env bash
# 2026-09-15.  Replaces chain_c10_waonly_20260915.sh, whose first cell (CIFAR-10 cfa_wa_eps8_100ep, pid 21802)
# had already started and keeps running; this waits for it.
# WA-only stack (no AWP): ARREST own recipe + WA at 8/255 on both datasets (20 epochs, cheap first),
# then CFA sample-wise eps + WA at 8/255 and 8.8/255.  CIFAR-100 8.8/255 already exists as
# ladder_angeps_wa_fh_100ep (identical to l2_bestrecipe_freezehead minus AWP).
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=21802
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') cfa_wa_eps8_100ep (CIFAR-10) finished ==="
for entry in CIFAR100:arrest_wa_20ep CIFAR10:arrest_wa_20ep \
             CIFAR100:cfa_wa_eps8_100ep CIFAR10:cfa_wa_eps88_100ep \
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
