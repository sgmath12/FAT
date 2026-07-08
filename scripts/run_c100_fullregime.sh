#!/bin/bash
# C100 FULL-regime (consist lamda=5 + SWA weight_avg=True) frontier check:
# does iso3's isolated +~2pp-clean edge over iso1 SURVIVE the boosters?
# Fair frontier: 3 alphas per cell. nonorm temp scale ~13, norm scale ~0.7.
# Appends to results/CIFAR100/full_nonorm_const/ and full_norm_const/output.log.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

run () { # cfg alpha
  echo "############ FULL $1  alpha=$2  START $(date) ############"
  $PY -u main.py --config_name "$1" --tau 0 --alpha "$2" --dataset CIFAR100
  echo "############ FULL $1  alpha=$2  DONE  $(date) ############"
}

# nonorm const full (alpha ~ raw scale, around iso1 best 13)
run full_nonorm_const.yaml  9
run full_nonorm_const.yaml  13
run full_nonorm_const.yaml  16
# norm const full (alpha ~ normed scale, around iso3 best 0.7)
run full_norm_const.yaml  0.5
run full_norm_const.yaml  0.7
run full_norm_const.yaml  1.0

echo "############ C100 FULLREGIME ALL DONE $(date) ############"
