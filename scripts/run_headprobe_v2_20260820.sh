#!/usr/bin/env bash
# ROBUST LINEAR PROBE v2 (2026-08-20).  Head refit on a frozen backbone, with the head INPUT divided
# by its own mean norm so the two designs face the same optimization problem.
#
# v1 did not: it handed a unit-norm-input head (direction) and a norm-12-input head (raw L2) the
# same lr and weight decay.  The directional probes plateaued near-uniform -- CE 3.48 and 4.00
# against ln(100) = 4.61 -- while beating the raw cells on train accuracy, i.e. flat logits, because
# reaching the same logit scale needs ~12x larger weights and weight decay fights exactly that.
#
# v2 therefore: (a) rescales the head input by a single global constant (no design property changes
# -- raw still delivers per-sample magnitude, direction still delivers none), (b) wd 0, (c) lr 0.1
# for 30 epochs.  With wd 5e-4 / lr 0.01 the directional head was still climbing after two epochs
# (|W| 5.97 -> 6.24) against the 11.88 its own trained head had reached.
#
# Reference, with each cell's originally trained head (C100 / ResNet18 / 100ep / seed 0, clean / AA):
#   bare      direction `b2x2_snorm_tnorm` 61.52 / 22.90   L2 `wadec_raw_nowa` 62.40 / 24.34
#   champion  direction `wadec_dir_full`   60.62 / 28.55   L2 `wadec_raw_full` 57.96 / 28.13
# v1 probe (confounded, kept for the record):     60.17 / 23.13   62.63 / 24.09
#                                                 52.86 / 26.20   60.12 / 27.81
set -u
cd "$(dirname "$0")/.."
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
mkdir -p logs

run () {
  echo "=== $(date '+%m-%d %H:%M') start probe2 $1 ==="
  $PY -u scripts/head_probe.py --cell "$1" \
      --ckpt "CIFAR100/checkpoint/$1/feat_direction_last.pkl" \
      --epochs 30 --lr 0.1 --wd 0 > "logs/probe2_$1.log" 2>&1
  echo "=== $(date '+%m-%d %H:%M') done probe2 $1 (exit $?) ==="
}

run b2x2_snorm_tnorm   # bare direction
run wadec_raw_nowa     # bare L2
run wadec_dir_full     # champion direction
run wadec_raw_full     # champion L2

echo "=== $(date '+%m-%d %H:%M') ALL DONE ==="
