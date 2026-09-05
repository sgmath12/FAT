#!/usr/bin/env bash
# REBUILD THE 50-EPOCH BLOCK OF tab:ladder IN THE SHIPPED DESIGN (2026-09-05).
#
# The 100-epoch block was rebuilt yesterday with featdir_freeze_head on.  That fixed one problem and
# created another: the two blocks of the table were then in different designs, and the table's claim
# that AWP changes sign with schedule length reads -0.25 NRR at 50 epochs against +1.08 at 100.  A
# sign change compared across mismatched blocks is confounded, so the 50-epoch block is rebuilt too.
#
# Four rungs rather than three, because unlike the 100-epoch block there is no existing freeze-head
# cell to sit at the top of this one.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in ladder_p0_fh_50ep ladder_angeps_fh_50ep ladder_angeps_wa_fh_50ep ladder_angeps_waawp_fh_50ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/CIFAR100_${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
