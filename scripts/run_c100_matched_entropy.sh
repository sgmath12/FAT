#!/bin/bash
# C100 matched-ENTROPY control: iso1 (no-norm const) vs iso3 (norm const) at the
# SAME teacher target-entropy. THE decisive test of whether iso3's C100 edge is the
# per-sample norm SHAPE (normalization equalizes/reshapes targets per sample) or just
# a global softness/temperature artifact.
#
# Prior confound: iso1-best (alpha~13, H=4.575) and iso3-best (alpha~0.7, H=4.542) were
# NOT entropy-matched, so part of iso3's win could be softness, not shape.
# Fix: pick alpha pairs giving EQUAL mean target entropy (scratchpad/c100_entropy_curve.py).
# If iso3 STILL beats iso1 at matched H -> normalization's per-sample shape genuinely helps
# (supports "norm-only, no kappa(x)" salvage). If they converge -> iso3 edge was softness.
#
# Sharper H (4.30) is most diagnostic: that's where the norm-driven target-shape differences
# bite hardest. Appends to results/CIFAR100/iso1_nonorm_const/ and iso3_norm_const/output.log.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

run () { # cfg alpha targetH
  echo "############ MATCHED-H $1  alpha=$2  (targetH=$3)  START $(date) ############"
  $PY -u main.py --config_name "$1" --tau 0 --alpha "$2" --dataset CIFAR100
  echo "############ MATCHED-H $1  alpha=$2  (targetH=$3)  DONE  $(date) ############"
}

# (iso1_alpha, iso3_alpha) per matched target entropy H:
#  H=4.30: 5.667 / 0.407   H=4.40: 6.512 / 0.442   H=4.48: 7.482 / 0.521
#  H=4.54: 9.877 / 0.667   H=4.58: 13.977 / 1.008
for pair in "5.667 0.407 4.30" "6.512 0.442 4.40" "7.482 0.521 4.48" "9.877 0.667 4.54" "13.977 1.008 4.58"; do
  set -- $pair
  run iso1_nonorm_const.yaml "$1" "$3"
  run iso3_norm_const.yaml   "$2" "$3"
done

echo "############ C100 MATCHED-ENTROPY ALL DONE $(date) ############"
