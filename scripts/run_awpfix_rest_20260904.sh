#!/usr/bin/env bash
# AWP-PROXY RE-RUNS, MINUS THE ONE ALREADY DONE (2026-09-04).
#
# Supersedes run_awpfix_rest_20260902.sh.  These cells were trained before the AWP proxy correction
# of app:awp, so their published numbers have to be reproduced post-fix; the paper states that every
# shipped-recipe number is post-correction, which makes this the queue's paper-critical block rather
# than a baseline sweep.
#
# CIFAR100/l2_bestrecipe_angeps is dropped: logs/CIFAR100_l2_bestrecipe_angeps_awpfix.log already
# exists from 09-02, and the old script had no skip logic, so it would have spent 2.2 h reproducing
# a run it already has.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
run () {
  if [ -s "logs/$1_$2_awpfix.log" ] && grep -ql "last_aa_acc" "results/$1"/*/"$2"/*.log 2>/dev/null; then
    echo "=== $(date '+%m-%d %H:%M') skip $1/$2 (awpfix log already present) ==="; return 0
  fi
  echo "=== $(date '+%m-%d %H:%M') start $1/$2 ==="
  $PY -u main.py --config_name "$2.yaml" --dataset "$1" --seed 0 > "logs/$1_$2_awpfix.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1/$2 (exit $?) ==="
}
run CIFAR100 ladder_angeps_waawp_100ep
run CIFAR100 ladder_angeps_waawp_50ep
run CIFAR100 champ_eps8
run CIFAR100 champ_eps10
run CIFAR10  champ_eps8
run CIFAR10  champ_eps10
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
