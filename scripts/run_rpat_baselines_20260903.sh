#!/usr/bin/env bash
# RPAT'S LEADERBOARD, RUN OURSELVES (2026-09-02).
#
# RPAT targets the same trade-off we do, so its Tables 3 and 4 are the field's own leaderboard for
# this problem.  We were quoting none of it.  Rather than quote, we run -- but only the cells that
# earn their GPU time.  What was skipped and why is recorded here so the choice is not silently
# re-litigated later:
#
#   RUN
#     WA alone            component of OUR stack; our "the stack is additive" claim currently rests
#                         on a PreActResNet-18 row from someone else's paper
#     WA + AWP            same, and the honest random-init counterpart to at_teacherinit_matched
#     Consistency-AT      strongest of RPAT's four benchmarks on clean (58.53 C100 vs PGD-AT 56.56),
#                         and it raises clean with no teacher at all -- the control our table needs
#     KD + SWA analogue   see run_kdswa_20260903.sh; the closest published method to ours structurally
#
#   SKIPPED
#     ReBAT               RPAT++ dominates it on both axes in RPAT's own Table 3 (56.84/27.68 against
#                         56.13/27.60) and we already reproduce RPAT++
#     TE                  no reference implementation available to us; implementing from the paper
#                         risks reporting a number that is our bug rather than their method
#     MMA, GAIRAT,        dominated on RPAT's own numbers -- C100 AA 18.40, 19.80, 16.70, 23.52, 24.30
#     MAIL, EWAT, SOVR    against our 28.86.  Running them would pad the table, not test anything.
#
# Protocol for every cell here is `pgdat_100ep`: SGD 0.1, piecewise decay at 70/90, 100 epochs, random
# init, eps 8/255.  That makes them comparable with our other standard baselines rather than with the
# published numbers, which are PreActResNet-18 and a different schedule.
#
# Consistency-AT doubles the batch before the attack, so budget roughly 2x a PGD-AT cell for it.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

for c in pgdat_wa_100ep pgdat_wa_awp_100ep consistency_100ep; do
  for ds in CIFAR100 CIFAR10; do
    [ -f "config/$ds/${c}.yaml" ] || { echo "skip $ds/$c (config 없음)"; continue; }
    echo "=== $(date '+%m-%d %H:%M') start $ds/$c ==="
    $PY -u main.py --config_name "${c}.yaml" --dataset "$ds" --seed 0 > "logs/${ds}_${c}.log" 2>&1
    echo "=== $(date '+%m-%d %H:%M') done $ds/$c (exit $?) ==="
  done
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
