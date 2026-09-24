#!/usr/bin/env bash
# 2026-09-24.  Moving the 100-epoch CIFAR-100 controls from 8.8/255 to the standard 8/255, which is the
# radius the reported results use.  The 8.8/255 cells cannot simply be relabelled, so each control is
# rerun at 8/255: the uniform-radius arm with and without the stack, the three allocation-signal controls
# (difficulty order, difficulty magnitude, logit margin), fixed-head logit regression, and the
# normalized-feature variant.  Runs behind the WA-by-AWP factorial.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
while pgrep -f 'main.py --config_name' > /dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') previous chain finished ==="
for cfg in eps8_p0_stack_100ep eps8_diffrank_100ep eps8_p0_nostack_100ep \
           eps8_diffmag_100ep eps8_margineps_100ep \
           eps8_logitmse_stack_100ep eps8_normfeat_stack_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
