#!/usr/bin/env bash
# LBGAT CIFAR-10, WHICH DIVERGED AND WAS RECORDED AS DONE (2026-09-04).
#
# results/CIFAR10/ResNet18/lbgat_100ep/2609030803.log ends at 10.0 on clean, FGSM, PGD-20/10/50, CW
# and AutoAttack alike -- chance on every metric, i.e. the run diverged.  run_lbgat_rerun_20260904
# then skipped the cell, because its completeness test was `grep -ql last_aa_acc`: a collapsed run
# writes that key like any other, so divergence reads as success.  That is the bug this replaces.
#
# The configuration is not the suspect.  config/CIFAR10/lbgat_100ep.yaml and its CIFAR-100 twin
# differ in the dataset name and the teacher checkpoint and nothing else, the CIFAR-100 cell trained
# to 56.46 / 26.02, and lr_schedule is already `lbgat` with its own decay points at 76 and 91.  So
# this is a re-run at the same settings rather than a repair, and if it diverges a second time the
# honest report is a failed reproduction, as with CURE, and not a number.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
# chance level, so a collapsed run is not mistaken for a finished one
CHANCE=10.0
best=$(grep -ho "'last_aa_acc': [0-9.]*" results/CIFAR10/*/lbgat_100ep/*.log 2>/dev/null \
       | grep -o "[0-9.]*$" | sort -g | tail -1)
if [ -n "$best" ] && awk "BEGIN{exit !($best > $CHANCE + 1.0)}"; then
  echo "=== $(date '+%m-%d %H:%M') skip CIFAR10/lbgat_100ep (AA $best already above chance) ==="
else
  echo "=== $(date '+%m-%d %H:%M') start CIFAR10/lbgat_100ep (best so far: ${best:-none}) ==="
  $PY -u main.py --config_name lbgat_100ep.yaml --dataset CIFAR10 --seed 0 > logs/CIFAR10_lbgat_100ep.log 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR10/lbgat_100ep (exit $?) ==="
fi
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
