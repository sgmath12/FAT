#!/usr/bin/env bash
# RE-RUN EVERY PAPER CELL AFFECTED BY THE AWP ASCENT FIX (2026-09-01).
#
# _awp_loss_fn diverged from _step_loss in two ways, so AWP searched for its weight perturbation using
# a loss the model does not train.  Measured on the champion: the head term was 7.15 against the
# anchor's 4.15 and supplied 40.8% of the AWP backbone gradient norm.  Fixed in 021f0ef.
#
# 21 configs are affected; these are the 9 that appear in the paper.  Cells WITHOUT awp_gamma never
# call _awp_loss_fn and are bit-identical under both versions, so the ladder's six non-AWP rows and
# every non-featdir baseline stay as they are.  Only the two AWP rows are redone, which leaves that
# ladder internally consistent rather than mixed.
#
# Ordered by what the paper needs first.  Logs are timestamped so every number is comparable
# before/after; the pre-fix values are:
#   CIFAR100 l2_bestrecipe_freezehead   62.65 / 28.77 / 39.43   <- headline
#   CIFAR10  champ_eps88                85.58 / 51.79 / 64.53   <- CIFAR-10 headline
#   CIFAR100 l2_bestrecipe_angeps       62.35 / 28.68 / 39.29   <- head-KD comparison row
#   CIFAR100 ladder_angeps_waawp_100ep  62.35 / 28.68 / 39.29
#   CIFAR100 ladder_angeps_waawp_50ep   59.90 / 28.05 / 38.21
#   CIFAR100 champ_eps8                 63.89 / 27.78 / 38.72
#   CIFAR100 champ_eps10                59.99 / 28.77 / 38.89
#   CIFAR10  champ_eps8                 87.22 / 51.15 / 64.48
#   CIFAR10  champ_eps10                83.29 / 51.79 / 63.87
#
# Tiny-ImageNet featdir_tin_100ep is affected too (freeze_head) and belongs to the other server,
# which now has the fix.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
run () {
  echo "=== $(date '+%m-%d %H:%M') start $1/$2 ==="
  $PY -u main.py --config_name "$2.yaml" --dataset "$1" --seed 0 > "logs/$1_$2_awpfix.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1/$2 (exit $?) ==="
}

run CIFAR10  adaadigdm_nat100ep_a1        # the alpha=1 IGDM control, approved separately
run CIFAR100 l2_bestrecipe_freezehead
run CIFAR10  champ_eps88
run CIFAR100 l2_bestrecipe_angeps
run CIFAR100 ladder_angeps_waawp_100ep
run CIFAR100 ladder_angeps_waawp_50ep
run CIFAR100 champ_eps8
run CIFAR100 champ_eps10
run CIFAR10  champ_eps8
run CIFAR10  champ_eps10
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
