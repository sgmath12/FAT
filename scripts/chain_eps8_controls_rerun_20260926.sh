#!/usr/bin/env bash
# 2026-09-26.  Relaunch of the seven 8/255 controls.  chain_eps8_controls_20260924.sh never started a
# single cell: it waits on `pgrep -f 'main.py --config_name'`, and the shell that CREATED it is still
# alive with that exact string in its own command line, so the guard matches itself and never clears.
# The GPU sat idle from 09-25 07:21 to 09-26 08:38 because of it.
#
# No pgrep guard here.  main.py takes an exclusive flock on /tmp/fat_gpu_0.lock, which is the real
# mutual exclusion; a pgrep on a command-line substring is not, as this failure shows.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for cfg in eps8_p0_stack_100ep eps8_diffrank_100ep eps8_p0_nostack_100ep \
           eps8_diffmag_100ep eps8_margineps_100ep \
           eps8_logitmse_stack_100ep eps8_normfeat_stack_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$cfg (exit $?) ==="
done
