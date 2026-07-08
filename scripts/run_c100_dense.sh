#!/bin/bash
# DENSE CIFAR100 isolated sweep — focus iso3 (norm const) vs iso4 (norm adapt).
# Motivation: unlike CIFAR10 (all 4 cells flat ~saturated), C100 iso3 shows a REAL
# TREND — alpha 0.3/0.5/0.7 -> pgd20 28.07/29.84/31.37, clean 61.7/62.2/63.5 (both
# still rising at 0.7, peak not found). So (a) push iso3 higher to locate the peak,
# and (b) trace iso4 as a 2D frontier (median temp x dispersion) at matched medians
# to test whether kappa(x) adaptivity adds ANYTHING over the const peak.
#
# C100 teacher raw-norm: median R=13.061, p5=10.485, p95=16.665 (disp 1.59x).
#   iso3 (norm const): T = alpha (median temp = alpha directly).
#   iso4 (norm adapt): T_i = tau*||Phi_i|| + alpha; median temp m = tau*R + alpha
#                      => alpha = m - tau*13.061. All points keep T>0 even at norm~8.
# Isolated: lamda=0, weight_avg=False already baked into the configs.
# Appends to results/CIFAR100/iso3_norm_const/output.log and iso4_norm_adapt/output.log.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

run () { # cfg tau alpha note
  echo "############ C100dense $1  tau=$2 alpha=$3  ($4)  START $(date) ############"
  $PY -u main.py --config_name "$1" --tau "$2" --alpha "$3" --dataset CIFAR100
  echo "############ C100dense $1  tau=$2 alpha=$3  DONE  $(date) ############"
}

# ---- iso3 norm const: find the peak (have 0.3/0.5/0.7, still rising) ----
run iso3_norm_const.yaml  0  0.8   "median_temp=0.8"
run iso3_norm_const.yaml  0  1.0   "median_temp=1.0"
run iso3_norm_const.yaml  0  1.3   "median_temp=1.3"
run iso3_norm_const.yaml  0  1.6   "median_temp=1.6"
run iso3_norm_const.yaml  0  2.0   "median_temp=2.0"

# ---- iso4 norm adapt: 2D frontier  median {0.7,1.0,1.3} x dispersion(tau) ----
run iso4_norm_adapt.yaml  0.03  0.308   "m=0.7 low-disp"
run iso4_norm_adapt.yaml  0.05  0.047   "m=0.7 mid-disp"
run iso4_norm_adapt.yaml  0.05  0.347   "m=1.0 mid-disp"
run iso4_norm_adapt.yaml  0.08 -0.045   "m=1.0 high-disp"
run iso4_norm_adapt.yaml  0.05  0.647   "m=1.3 mid-disp"
run iso4_norm_adapt.yaml  0.10 -0.006   "m=1.3 high-disp"

echo "############ C100 DENSE ALL DONE $(date) ############"
