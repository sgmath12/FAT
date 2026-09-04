#!/usr/bin/env bash
# REBUILD tab:ladder IN THE SHIPPED DESIGN (2026-09-04).
#
# The ladder measured each component with the classifier head trained, then added "freeze the head"
# as a final rung.  Two things are wrong with that.  Every rung is a measurement in a design the paper
# does not ship, so the deltas it reports -- what sensitivity-matched eps buys, what WA buys, what AWP
# buys -- are deltas in the wrong design.  And the head, which the method section says is inherited and
# never trained, appears as a component rather than as part of the construction.
#
# These three rungs are the same cells with featdir_freeze_head on.  The fourth is already in hand:
# l2_bestrecipe_freezehead, 62.17 / 32.37 / 30.93 / 28.86 / 39.42, which is the row tab:main reports,
# so the rebuilt ladder ends where the main table begins.
#
# It also settles the AWP question that cost this queue an argument.  The old rungs predate the
# AWP-proxy correction of app:awp; these are trained after it, so the rebuilt ladder is uniformly
# post-correction -- 6.6 h, against the 12 h of re-runs that would have left the design mismatch in
# place.
#
# 50 epochs is not rebuilt.  100 epochs is the schedule the paper ships and there is no 50-epoch
# freeze-head cell to end that block at; the caption says so.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in ladder_p0_fh_100ep ladder_angeps_fh_100ep ladder_angeps_wa_fh_100ep; do
  echo "=== $(date '+%m-%d %H:%M') start CIFAR100/$c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/CIFAR100_${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR100/$c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
