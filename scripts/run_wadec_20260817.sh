#!/usr/bin/env bash
# WA DECOMPOSITION (2026-08-17). Isolate WA out of the champion stack (WA + AWP proxy +
# train_eps 8.8/255 + freeze_lr_epoch): does WA ALONE flip the direction-vs-raw order?
#
# no-stack     -> raw wins AA  (C10 50ep full raw +0.41, C100 50ep +0.25)
# +full stack  -> direction wins (C100 +0.65, C10 +0.45, both at angeps p=0)
#
# 2x2 per dataset: {direction, FULL raw} x {no WA, WA}, 100ep, angeps OFF, AWP/8.8/freeze stripped.
# C100 direction+noWA already exists as b2x2_snorm_tnorm (61.52 / PGD 26.92 / CW 24.48 / AA 22.90),
# so 7 runs, not 8.  ~1.8h each => ~13h.
#
# Reference cells for reading the result (100ep, angeps p=0):
#   full stack  C100 dir 60.74 / AA 28.69   raw 57.78 / AA 28.04
#   full stack  C10  dir 82.52 / AA 51.89   raw 81.55 / AA 51.44
# WA-only lands somewhere between these and the no-WA row; the question is whether the ORDER has
# already flipped by the time only WA is on.
#
# NOTE: this is the first batch run after the main.py `_last.pkl` fix (2026-08-17) -- checkpoints
# saved here are the true final-epoch models, unlike everything saved before it.
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {  # run <dataset> <config_name>
  local ds="$1" c="$2"
  echo "=== $(date '+%m-%d %H:%M') start ${ds}/${c} ==="
  $PY -u main.py --config_name "${c}.yaml" --dataset "$ds" --seed 0 \
      > "logs/wadec_${ds}_${c}.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done ${ds}/${c} (exit $?) ==="
}

# C100 first: it is the dataset the flip was originally observed on, and its no-WA direction
# control already exists, so the 2x2 closes after three runs.
run CIFAR100 wadec_raw_nowa
run CIFAR100 wadec_dir_wa
run CIFAR100 wadec_raw_wa

run CIFAR10  wadec_dir_nowa
run CIFAR10  wadec_raw_nowa
run CIFAR10  wadec_dir_wa
run CIFAR10  wadec_raw_wa

echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
