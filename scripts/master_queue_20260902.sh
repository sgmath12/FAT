#!/usr/bin/env bash
# SINGLE SEQUENTIAL QUEUE -- ABLATIONS FIRST (2026-09-02, revised).
#
# This machine does ablations; Tiny-ImageNet and WideResNet are filled elsewhere.  The AWP-fix
# re-runs were ahead of the ablations and have been demoted to last: measured on the champion, the fix
# moves NRR by 0.01 (39.43 -> 39.42), so the remaining seven cells are decimal updates.  The ablations
# answer questions the paper currently cannot -- whether the anchor buys robustness by itself, whether
# the attack or the loss earns it, whether the read point matters, and whether p = 1 is a plateau.
#
# One process, one loop, no polling: the earlier design used independent waiters and put two trainings
# on one GPU.
set -u
cd "$(dirname "$0")/.."
while pgrep -f "main.py --config_name champ_eps88" > /dev/null; do sleep 120; done
echo "=== $(date '+%m-%d %H:%M') champ_eps88 done ==="
for q in run_ablations_20260901 run_std_baselines_20260902 run_lowerps_20260902 run_awpfix_rest_20260902; do
  echo "=== $(date '+%m-%d %H:%M') >>> $q ==="
  bash "scripts/$q.sh"
  echo "=== $(date '+%m-%d %H:%M') <<< $q done ==="
done
echo "=== $(date '+%m-%d %H:%M') MASTER QUEUE DONE ==="
