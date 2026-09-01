#!/usr/bin/env bash
# THE ABLATIONS THE CLAIMS ACTUALLY NEED (2026-09-01).  See writting_docs/paper/notes/ablation_plan.md
# for the full map of claim -> evidence -> status, built from an inventory of all 85 completed runs.
#
# Every cell here is the BASE REGIME -- 50 epochs, no weight averaging, no AWP, eps 8/255 -- except the
# scratch cell, which must match ladder_p0_100ep.  That is deliberate: an ablation measured inside the
# full stack cannot be attributed, because WA and AWP both move the same axis.
#
# Ordered by how much of the paper rests on the answer.
#
#   1  abl_teacher_at_adv   the CENTRAL claim: teacher read at x_adv instead of x.
#                           vs ladder_p0_50ep  61.33 / AA 26.19 / NRR 36.71
#   2-4 abl_ce_lam{01,03,10} "the anchor replaces cross-entropy rather than supplementing it"
#                           brackets: pure CE from teacher weights 57.73/26.46, anchor alone 61.33/26.19
#   5-6 abl_angeps_p{05,20} makes "no hyperparameter" a measured plateau; p=0 and p=1 already exist
#   7  abl_scratch_init     separates initialization from the anchor, at the shipped 100-epoch recipe.
#                           vs ladder_p0_100ep 61.21 / 25.24
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs
run () {
  echo "=== $(date '+%m-%d %H:%M') start $1 ==="
  $PY -u main.py --config_name "$1.yaml" --dataset CIFAR100 --seed 0 > "logs/CIFAR100_$1.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done $1 (exit $?) ==="
}
run abl_teacher_at_adv
run abl_ce_attack
run abl_ce_lam03
run abl_ce_lam01
run abl_ce_lam10
run abl_angeps_p05
run abl_angeps_p20
run abl_scratch_init
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
