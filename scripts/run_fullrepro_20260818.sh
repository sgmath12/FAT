#!/usr/bin/env bash
# STACK DECOMPOSITION step 4 / reproduction check (2026-08-18).
# The champion-regime row is the only row of the chain not produced by this chain: it comes from
# runs of 2026-08-01 / 08-02, before the angeps, adaptive-pooling and design-axis commits.  Its L2
# cell is the suspicious one -- freeze_lr appears to cost L2 -2.43 clean against direction's -0.67,
# and that asymmetry is what flips the sign.  These two cells re-run it on today's code.
# Expect (if the old rows are sound): dir 60.74 / AA 28.69, L2 57.78 / AA 28.04.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in wadec_raw_full wadec_dir_full; do   # L2 first: it is the cell under suspicion
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
