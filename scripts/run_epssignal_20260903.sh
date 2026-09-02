#!/usr/bin/env bash
# WHICH PER-SAMPLE RADIUS SIGNAL (2026-09-03).
#
# Section 3.2 distinguishes our radius rule from the existing per-sample-eps family (IAAT, MMA, CAT)
# by WHAT it allocates from: the input-sensitivity of the training loss, not the sample's difficulty
# or its input-space margin.  The paper currently supports that with one control, `champ_diffrank`,
# which keeps our own weights and merely reorders them by difficulty.  That is a permutation test, not
# a comparison of signals -- the other two cells were never run.
#
# Reimplementing IAAT, MMA and CAT end to end would compare whole recipes rather than allocation
# rules, so these swap the signal in place: same exponent p, same [0.5, 1.5] clamp, same mean
# restoration, therefore the SAME TOTAL ATTACK BUDGET, and nothing else touched.
#
#   champ_diffeps     difficulty magnitude (per-sample CE) -- the IAAT / CAT direction
#   champ_margineps   logit margin, inverted               -- MMA's criterion
#
# Read against the champion 62.17 / 28.86 and champ_diffrank 61.52 / 27.95.  These belong in the
# appendix beside tab:ablation, which is where the claim is made.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
for c in champ_diffeps champ_margineps; do
  echo "=== $(date '+%m-%d %H:%M') start $c ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset CIFAR100 --seed 0 > "logs/CIFAR100_${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $c (exit $?) ==="
done
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
