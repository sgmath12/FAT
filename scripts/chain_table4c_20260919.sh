#!/usr/bin/env bash
# 2026-09-19 09:40.  Replaces chain_table4b_20260919.sh: the three cells Table 4 is still missing and
# nothing else.  The natural-init-only cells were dropped too -- tab:natteacher carries the four
# distillation objectives alone, so Consistency-AT, LBGAT and ADR have no row there.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=76590
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
echo "=== $(date '+%m-%d %H:%M') adaadigdm_natinit_stack_100ep finished ==="
for cfg in consistency_natinit_stack_100ep lbgat_natinit_stack_100ep adr_natinit_stack_200ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
