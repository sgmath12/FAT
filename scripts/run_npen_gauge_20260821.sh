#!/usr/bin/env bash
# NORM-GAUGE TEST (2026-08-21).  Is the directional design's edge the fact that it leaves the
# feature norm free, and spends that free coordinate as an implicit LR dial on the angle?
#
# Measured (clean test, teacher ||Phi_t|| = 11.22):
#   direction ||Phi_s||: bare 20.29 -> +WA 22.99 -> +AWP 26.79 -> +freeze_lr 36.22   (1.81x -> 3.23x)
#   raw L2    ||Phi_s||: bare  9.39 -> +WA  8.84 -> +AWP  8.48 -> +freeze_lr  9.26   (pinned ~0.8x)
# The L2 cells sit on the value their own loss commands, ||Phi_t||cos(theta) = 9.19-9.43, in every
# regime; the directional ones float and grow most exactly where the LR stops decaying (+30% vs
# +2.6%).  Since L_dir sees Phi only through Phi_hat, dL/dPhi = (1/||Phi||)(I - PP^T) dL/dPhi_hat,
# so a 3x norm is a 3x smaller angular step: growing the norm IS annealing.
#
# THE READ IS THE freeze_lr DELTA, not the absolute numbers.  npen does two things at once -- it
# removes the norm freedom AND adds a magnitude-matching objective (precisely the extra constraint
# the L2 design carries) -- so npen cells may be worse outright.  That second effect is held fixed
# inside the pair, so the delta is clean.
#
#   without npen:  +AWP 61.41 / 28.07  ->  + freeze_lr 60.62 / 28.55    (AA +0.48, clean -0.79)
#   with npen:     prediction -- the +0.48 shrinks toward zero or reverses, because the dial is welded.
#
# If it holds, theory_v1 T.6's empty slot gets a design-intrinsic mechanism instead of a schedule
# interaction: leaving a decision-null coordinate unconstrained lets the optimizer self-anneal.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

for c in wadec_dir_wa_eps88_awp_npen wadec_dir_full_npen; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
