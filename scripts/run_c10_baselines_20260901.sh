#!/usr/bin/env bash
# CIFAR-10 COUNTERPARTS OF THE NATURAL-TEACHER BASELINE TABLE (2026-09-01).
#
# CIFAR-100 already has all of this; CIFAR-10 has none of it, so the analysis section's claim --
# given a natural teacher, every published distillation objective lands below simply not distilling --
# currently rests on one dataset.  Six cells fix that.  Every config is the CIFAR-100 one with the
# dataset and the teacher checkpoint changed and nothing else, so the two datasets are read on the
# same footing.
#
# Order: the "do not distil" reference first, because it is the row everything else is compared to and
# the table is unreadable without it; then the baselines cheapest-first; then our stack-free anchor.
#
# CIFAR-100 results for reference:
#   PGD-AT @ teacher init  57.73 / AA 26.46      anchor p=0, no stack  61.21 / AA 25.24
#   ARD 57.61 / 20.24    RSLAD 59.68 / 21.30    AdaAD 59.79 / 23.19    AdaAD+IGDM(a=20) COLLAPSED
#
# One thing differs by construction and is not a porting error: RSLAD's `10 * mean(elementwise KL)`
# is exactly the batch mean when C = 10, while on CIFAR-100 the same constant divided the only loss
# term by ten.  Same config value, different effective scale -- see train_rslad's docstring.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {
  echo "=== $(date '+%m-%d %H:%M') start CIFAR10/$1 ==="
  $PY -u main.py --config_name "$1.yaml" --dataset CIFAR10 --seed 0 > "logs/CIFAR10_$1.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done CIFAR10/$1 (exit $?) ==="
}

run at_teacherinit_matched
run ard_nat100ep
run rslad_nat100ep
run adaad_nat100ep
run adaadigdm_nat100ep
run ladder_p0_100ep
echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
