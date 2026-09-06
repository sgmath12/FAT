#!/usr/bin/env bash
# The rest of the baselines with our recipe AND our stack (waits on PID 1688570).
# ARD, RSLAD and AdaAD+IGDM, so that every published objective in tab:main has been run with
# everything our own row uses.  AdaAD's twin is already queued ahead of these.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
PID=1688570
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
for cfg in ard_ourrecipe_wa_awp_100ep rslad_ourrecipe_wa_awp_100ep adaadigdm_ourrecipe_wa_awp_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
