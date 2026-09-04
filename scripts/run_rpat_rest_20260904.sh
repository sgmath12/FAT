#!/usr/bin/env bash
# THE NON-ADR HALF OF run_rpat_baselines_20260903 (2026-09-04).
#
# adr_200ep is not repeated here: CIFAR100 finished at 13:14 (AA 27.20) and CIFAR10 is on the card
# as this is written.  Re-running the original script would have redone both, at 4 h each.
#
# These are table-filling baselines for tab:main, which is why they sit at the end of the queue
# rather than the front.  pgdat/trades/mart are NOT here -- they are being run on another machine.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in pgdat_wa_100ep pgdat_wa_awp_100ep consistency_100ep; do
  for ds in CIFAR100 CIFAR10; do
    [ -f "config/$ds/${c}.yaml" ] || { echo "skip $ds/$c (config 없음)"; continue; }
    echo "=== $(date '+%m-%d %H:%M') start $ds/$c ==="
    $PY -u main.py --config_name "${c}.yaml" --dataset "$ds" --seed 0 > "logs/${ds}_${c}.log" 2>&1
    echo "=== $(date '+%m-%d %H:%M') done $ds/$c (exit $?) ==="
  done
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
