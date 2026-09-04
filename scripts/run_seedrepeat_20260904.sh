#!/usr/bin/env bash
# HOW BIG IS A DIFFERENCE (2026-09-04).
#
# Every number in this paper is one seed, which app:repro states.  That is tolerable while the text
# only reports differences it does not interpret, but it stops being tolerable the moment we want to
# say a gap is small: the L1-vs-L2 allocation lands 0.29 AutoAttack apart, several ablation rows are
# read as ties at similar distances, and nothing in the paper says what seed-to-seed spread looks
# like on this recipe.
#
# This is the shipped CIFAR-100 cell at seed 1.  Against seed 0's 62.17 / 32.37 / 30.93 / 28.86 it
# gives the one number the paper is missing, and it costs one cell.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
echo "=== $(date '+%m-%d %H:%M') start CIFAR100/l2_bestrecipe_freezehead seed 1 ==="
$PY -u main.py --config_name l2_bestrecipe_freezehead.yaml --dataset CIFAR100 --seed 1 \
    > logs/CIFAR100_l2_bestrecipe_freezehead_seed1.log 2>&1
echo "=== $(date '+%m-%d %H:%M') done (exit $?) ==="
