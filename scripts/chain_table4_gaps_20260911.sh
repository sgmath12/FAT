#!/usr/bin/env bash
# 2026-09-11.  The holes in tab:main's three layers, found by auditing all thirteen baselines rather
# than the seven that happened to have cells.  PGD-AT at its own recipe from a random init and
# Consistency-AT had never been run here at all; LBGAT and ADR had no warm-start cell; Consistency and
# ADR had no full-match cell.  ARREST and RPAT++ need code, not configs, so they are not here.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=17118
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in pgdat_100ep consistency_100ep consistency_natinit_100ep consistency_full_100ep \
           lbgat_natinit_100ep adr_natinit_200ep adr_full_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
