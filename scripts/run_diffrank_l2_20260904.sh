#!/usr/bin/env bash
# PERMUTATION CONTROL, RE-RUN ON THE SHIPPED RECIPE (2026-09-04).
#
# The control the paper cites in Section 3 (`champ_diffrank`, 2026-08-22) was matched to the
# then-champion `featdir_champ200_angeps`: student_norm True (directional / partial raw) and
# freeze_lr_epoch 0.65.  The shipped recipe is `l2_bestrecipe_freezehead` -- raw L2 anchor, head
# frozen, no freeze_lr -- so the control was measuring the old design.  Both cells here are
# byte-identical to the shipped config except for one line each.
#
#   champ_p0_l2       featdir_angeps_p    = 0.0                -> uniform row
#   champ_diffrank_l2 featdir_eps_signal  = difficulty_rank    -> permuted row
#
# The ours row already exists: l2_bestrecipe_freezehead, 62.17 / 32.37 / 30.93 / 28.86 / 39.42.
#
# Also read `gradnorm_cv` out of the batch-0 log line of the diffrank cell: the 0.03-vs-0.64
# dispersion figure quoted in Section 3 was measured on a unit-norm student and may not survive
# the move to a raw one (methods.py:2500).
#
# Safe to launch at any time: main.py holds a per-GPU flock and WAITS, so this queues behind
# whatever is on the card rather than racing it.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in champ_p0_l2 champ_diffrank_l2; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/CIFAR100_${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
