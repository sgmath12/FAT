#!/usr/bin/env bash
# THE REST OF THE AWP-FIX RE-RUNS -- LOW PRIORITY (2026-09-02).
#
# Deferred behind the ablations after measuring what the fix actually changes.  The CIFAR-100 champion
# went 62.65 / 28.77 / NRR 39.43 before the fix to 62.17 / 28.86 / 39.42 after: NRR moved by 0.01.
# AWP is evidently insensitive to the 40.8%% of its ascent direction that came from the head term, so
# these seven cells are decimal-place updates costing about sixteen hours.  The paper's description now
# matches the code either way, which was the reason for the fix.
#
# Run when the ablations and the standard baselines are done, and update the tables then.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run () {
  echo "=== $(date '+%m-%d %H:%M') start $1/$2 ==="
  $PY -u main.py --config_name "$2.yaml" --dataset "$1" --seed 0 > "logs/$1_$2_awpfix.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1/$2 (exit $?) ==="
}
run CIFAR100 l2_bestrecipe_angeps
run CIFAR100 ladder_angeps_waawp_100ep
run CIFAR100 ladder_angeps_waawp_50ep
run CIFAR100 champ_eps8
run CIFAR100 champ_eps10
run CIFAR10  champ_eps8
run CIFAR10  champ_eps10
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
