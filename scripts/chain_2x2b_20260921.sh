#!/usr/bin/env bash
# 2026-09-21.  Revised: one logit-MSE cell rather than a logit column, and the repeat budget spent on the
# radius question instead.  Order:
#   1. fixed-head logit MSE with the sensitivity-matched radius, final recipe, one run.  Expected to tie
#      with the feature cell; if it does, that is reported as evidence that what matters is the fixed
#      clean target rather than a feature-specific metric.
#   2. CFA uniform against sensitivity-matched at seeds 1 and 2, which is the claim that has to repeat:
#      +2.35 clean at seed 0 (62.40/27.84 against 64.75/27.98).
#   3. the table-hole cells the 2x2 pushed back.
# A frozen-head teacher-only KL cell with the same stack and radius is the next experiment, not this one.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=100643
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') previous cell finished ==="
run() { echo "=== $(date '+%m-%d %H:%M') start $1 seed $2 ==="; \
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed $2 > logs/CIFAR100_$1_s$2.log 2>&1; \
  echo "=== $(date '+%m-%d %H:%M') done $1 seed $2 (exit $?) ==="; }
run logitmse_eps8_150ep_p1 0
run cfa_eps8_150ep_p0 1
run cfa_eps8_150ep 1
run cfa_eps8_150ep_p0 2
run cfa_eps8_150ep 2
run cfa_nostack_eps8_100ep 0
for cfg in ard_natinit_100ep rslad_natinit_100ep adaad_natinit_100ep adaadigdm_natinit_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR10/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR10 --seed 0 > logs/CIFAR10_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR10/$cfg (exit $?) ==="
done
