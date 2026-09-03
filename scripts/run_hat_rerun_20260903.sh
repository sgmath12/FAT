#!/usr/bin/env bash
# HAT RE-RUN ON ITS OWN SCHEDULE (2026-09-03).
#
# The first pass ran HAT under our baseline protocol -- SGD 0.1, piecewise, no warmup -- and it
# collapsed to chance from step 0 on both datasets: 1.00 on CIFAR-100 and 9.99 on CIFAR-10, pinned
# there for every evaluation.  Those logs are kept as `hat_100ep_collapsed_flatlr` rather than deleted.
#
# HAT's own trainer uses OneCycleLR at lr 0.21 with pct_start 0.25, which starts near zero and ramps.
# That warmup is evidently what its TRADES-KL plus helper-CE objective needs, and forcing our flat
# schedule on it produced our optimizer failure rather than their method -- the same rule that made us
# skip TE and re-run LBGAT on its own schedule.
#
# Safe to launch at any time: main.py holds a per-GPU flock and WAITS, so this queues behind whatever
# is on the card instead of racing it.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for ds in CIFAR100 CIFAR10; do
  echo "=== $(date '+%m-%d %H:%M') start $ds/hat_50ep (HAT schedule) ==="
  $PY -u main.py --config_name hat_50ep.yaml --dataset "$ds" --seed 0 > "logs/${ds}_hat_50ep.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $ds/hat_50ep (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
