#!/usr/bin/env bash
# ADR AT ITS BEST PUBLISHED CONFIGURATION (2026-09-04).
#
# adr_200ep is weight_avg False, i.e. AT + AWP + ADR, which ADR does not report.  The row the paper
# quotes and the row component-matched to our recipe is AT + WA + AWP + ADR (57.36 / 28.50 on
# CIFAR-100).  These two cells are adr_200ep with WA turned on and nothing else changed.
#
# The WA-off runs are kept rather than discarded: CIFAR-100 finished at 13:14 with 59.93 / 27.20 and
# CIFAR-10 is on the card, and together they are a matched pair showing what WA is worth to ADR.
# Both configurations get labelled for what they are in tab:main -- neither is "ADR" unqualified.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for ds in CIFAR100 CIFAR10; do
  echo "=== $(date '+%m-%d %H:%M') start $ds/adr_wa_200ep ==="
  $PY -u main.py --config_name adr_wa_200ep.yaml --dataset "$ds" --seed 0 > "logs/${ds}_adr_wa_200ep.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $ds/adr_wa_200ep (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
