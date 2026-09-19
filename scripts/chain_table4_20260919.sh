#!/usr/bin/env bash
# 2026-09-19 09:20.  Replaces chain_bench_20260917.sh.  The four cells Table 4 is missing come first;
# the CIFAR-10 RPAT++ natural-init run it was on appears in no table and is dropped.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run_fat() {
  DS="$1"; cfg="$2"
  echo "=== $(date '+%m-%d %H:%M') start $DS/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset $DS --seed 0 > logs/${DS}_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $DS/$cfg (exit $?) ==="
}
for entry in CIFAR100:adaadigdm_natinit_stack_100ep CIFAR100:consistency_natinit_stack_100ep \
             CIFAR100:lbgat_natinit_stack_100ep CIFAR100:adr_natinit_stack_200ep \
             CIFAR100:consistency_natinit_100ep CIFAR100:lbgat_natinit_100ep \
             CIFAR100:adr_natinit_200ep \
             CIFAR100:consistency_full_100ep CIFAR100:adr_full_100ep; do
  run_fat "${entry%%:*}" "${entry##*:}"
done
