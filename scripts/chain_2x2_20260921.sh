#!/usr/bin/env bash
# 2026-09-21.  The 2x2 the reviewer question turns on: {feature MSE, fixed-head logit MSE} x {uniform,
# sensitivity-matched radius}, all at the final recipe -- 200-epoch teacher, teacher initialization,
# frozen head, 150 epochs, 8/255, WA and AWP -- each loss generating its own attack and, where the radius
# is sample-wise, allocating it from its own input sensitivity (logitmse_angeps_p / featdir_angeps_p).
# Feature seed 0 is already measured: uniform 62.40/27.84, sample-wise 64.75/27.98.  So the logit pair at
# seed 0 comes first, then all four at seed 1 and at seed 2, so that the feature-logit and the
# sample-wise-uniform differences can be read WITHIN a seed rather than across means.
# Note: with a constant learning rate of 0.021 the logit cell diverges within twelve steps under the
# sample-wise radius; under the one-cycle schedule it starts far lower.  Watch the first epochs.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=100643
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') previous cell finished ==="
run() { echo "=== $(date '+%m-%d %H:%M') start $1 seed $2 ==="; \
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed $2 > logs/CIFAR100_$1_s$2.log 2>&1; \
  echo "=== $(date '+%m-%d %H:%M') done $1 seed $2 (exit $?) ==="; }
run logitmse_eps8_150ep_p0 0
run logitmse_eps8_150ep_p1 0
for s in 1 2; do
  run cfa_eps8_150ep_p0 $s
  run cfa_eps8_150ep $s
  run logitmse_eps8_150ep_p0 $s
  run logitmse_eps8_150ep_p1 $s
done
# the table-hole cells the 2x2 pushed back
run cfa_nostack_eps8_100ep 0
for cfg in ard_natinit_100ep rslad_natinit_100ep adaad_natinit_100ep adaadigdm_natinit_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR10/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR10 --seed 0 > logs/CIFAR10_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR10/$cfg (exit $?) ==="
done
