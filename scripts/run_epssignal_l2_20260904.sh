#!/usr/bin/env bash
# WHICH PER-SAMPLE RADIUS SIGNAL -- ON THE SHIPPED RECIPE (2026-09-04).
#
# Replaces run_epssignal_20260903.sh, whose two cells (champ_diffeps, champ_margineps) were built
# against the then-champion featdir_champ200_angeps: student_norm True (directional / partial raw)
# plus freeze_lr_epoch 0.65.  The shipped recipe is l2_bestrecipe_freezehead, and Section 3 -- where
# the claim is made -- describes the shipped rule, so the controls have to be measured there.
#
#   champ_diffeps_l2    difficulty magnitude (per-sample CE) -- the IAAT / CAT direction
#   champ_margineps_l2  logit margin, inverted               -- MMA's criterion
#
# Both are byte-identical to l2_bestrecipe_freezehead but for featdir_eps_signal.  Read against the
# champion 62.17 / 32.37 / 30.93 / 28.86 / 39.42 and against champ_p0_l2 (uniform) and
# champ_diffrank_l2 (permutation), which run first in run_diffrank_l2_20260904.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in champ_diffeps_l2 champ_margineps_l2; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/CIFAR100_${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
