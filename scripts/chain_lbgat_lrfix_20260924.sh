#!/usr/bin/env bash
# 2026-09-24.  LBGAT under the authors' learning-rate schedule, which we had transcribed one epoch late.
#
# Their adjust_learning_rate (train_lbgat_cifar100.py:152) is 1-indexed and runs BEFORE each epoch, so
# their first epoch trains at 0.02 and the decays land on epochs 76 and 91.  Ours compared their epoch
# thresholds against a 0-based fractional epoch, which put the warmup on the second epoch and both
# decays one epoch late, leaving step 0 at the full 0.1.  LBGAT's target term is nn.MSELoss(), a mean
# over batch AND classes, so a ten-class problem sees ten times the gradient of a hundred-class one at
# equal per-logit error: CIFAR-100 survived the shift, CIFAR-10 sat at chance from step 0 in both
# published variants (beta = 6 and beta = 0), and tab:main's CIFAR-10 LBGAT cell was left empty.
#
# Both datasets are rerun under the corrected schedule.  CIFAR-100 is included because its published
# number (56.46 / 26.02) was produced under the shifted schedule, and the two cells have to share a
# recipe.  The pre-fix logs are kept as logs/*_lbgat*.prelrfix.log.
#
# Then the two CIFAR-10 cells of the WA-by-AWP factorial: chain_2x2stack died at 17:45 with
# cfa_nostack_eps8_100ep mid-run, and cfa_awponly_eps8_100ep never started.
#
# main.py holds an exclusive flock on /tmp/fat_gpu_0.lock, so this chain and the still-armed
# chain_eps8_controls cannot put two jobs on the GPU at once.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run() { echo "=== $(date '+%m-%d %H:%M') start $2/$1 ==="; \
  $PY -u main.py --config_name $1.yaml --dataset $2 --seed 0 > logs/$2_$1.log 2>&1; \
  echo "=== $(date '+%m-%d %H:%M') done $2/$1 (exit $?) ==="; }
run lbgat0_100ep            CIFAR10    # the variant the authors publish for CIFAR-10 (beta = 0)
run lbgat_100ep             CIFAR100   # LBGAT6, re-measured under the corrected schedule
run cfa_nostack_eps8_100ep  CIFAR10
run cfa_awponly_eps8_100ep  CIFAR10
