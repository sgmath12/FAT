#!/usr/bin/env bash
# 2026-09-11 17:50.  Re-queued after finding my cells starving on the GPU lock: trades_full_100ep sat
# 6h35m printing "다른 학습이 사용 중" while another session's chain held it, because the 60-second
# poll always loses to a process that is already running.  Two chains competing for one lock is the
# trap recorded in memory; this one waits on the other chain's PID instead.
#
# The five sensitivity cells (ship_lr*, ship_ep200) are dropped: the other chain owns them and had
# already finished three of them, so keeping them here would have run 13 hours of duplicates.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
OTHER=12818
while kill -0 "$OTHER" 2>/dev/null; do sleep 60; done
echo "=== $(date '+%m-%d %H:%M') other chain drained, starting mine ==="
for cfg in trades_full_100ep mart_full_100ep hat_full_100ep lbgat_full_100ep \
           pgdat_100ep consistency_100ep consistency_natinit_100ep consistency_full_100ep \
           lbgat_natinit_100ep adr_natinit_200ep adr_full_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$cfg ==="
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 > logs/CIFAR100_${cfg}_s0.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $cfg (exit $?) ==="
done
