#!/usr/bin/env bash
# 2026-09-13.  Relaunch after the 04:06 reboot, nothing else running.
#  1. ARREST + our per-sample radius (arrest_angeps_p 1.0) on its own recipe and on the full match,
#     both datasets -- does ARREST catch or pass CFA once it has the one component it lacks?
#  2. Own recipe + natural init + stack (WA, AWP with warmup at 10% of epochs), cheap first.
#  3. Consistency-AT reruns: _jensen_shannon_div now floors its KL targets (collapse was the NaN).
#  4. Leftover layer-2 / layer-3 holes.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for entry in CIFAR100:arrest_angeps_20ep CIFAR10:arrest_angeps_20ep \
             CIFAR100:arrest_stack_20ep CIFAR100:hat_natinit_stack_50ep \
             CIFAR100:arrest_angeps_full_100ep CIFAR10:arrest_angeps_full_100ep \
             CIFAR100:consistency_100ep \
             CIFAR100:pgdat_natinit_stack_100ep CIFAR100:trades_natinit_stack_100ep \
             CIFAR100:mart_natinit_stack_100ep CIFAR100:ard_natinit_stack_100ep \
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
