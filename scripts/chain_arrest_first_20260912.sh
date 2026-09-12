#!/usr/bin/env bash
# 2026-09-12.  ARREST first, on both datasets: it is the closest published method to ours by
# construction -- natural warm start plus a representation-distance term -- and tab:main has it empty
# on CIFAR-10 and CIFAR-100 alike, with not even a quoted number.  Two cells per dataset: their own
# recipe (SGD, 20 finetuning epochs, lr 0.025 halving after 11, lambda 50, Noisy Replay at 30 degrees
# over the first half) and the full-match cell with our optimizer, radius and stack.
#
# Then the audited holes that were already queued.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=23934
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for entry in CIFAR100:arrest_20ep CIFAR10:arrest_20ep CIFAR100:arrest_full_100ep CIFAR10:arrest_full_100ep \
             CIFAR100:consistency_100ep CIFAR100:consistency_natinit_100ep \
             CIFAR100:consistency_full_100ep CIFAR100:lbgat_natinit_100ep \
             CIFAR100:adr_natinit_200ep CIFAR100:adr_full_100ep; do
  DS="${entry%%:*}"; cfg="${entry##*:}"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
done
