#!/bin/bash
# SEARCH for a non-normalization adaptive-temperature signal matching/beating normalization.
# nonorm base, isolated, alpha=13. shat clamped [0.5,2] (matched mild dispersion).
# uniform == iso1 baseline (60.71/31.04/25.82); norm == normalization control.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run () { # signal alpha
  echo "############ ADATEMP signal=$1 alpha=$2  START $(date) ############"
  FAT_ADASIGNAL="$1" $PY -u main.py --config_name adatemp.yaml --alpha "$2" --dataset CIFAR100
  echo "############ ADATEMP signal=$1 alpha=$2  DONE  $(date) ############"
}
# pass 1: all signals at alpha=13 (matched mean temp)
for s in uniform norm margin maxprob dev; do run "$s" 13; done
# pass 2: the novel signals at two more alphas (frontier)
for s in dev margin maxprob; do run "$s" 9; run "$s" 16; done
echo "############ C100 ADATEMP ALL DONE $(date) ############"
