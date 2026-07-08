#!/bin/bash
# DENSE extension of the isolated 2x2 (runs AFTER run_iso_2x2.sh).
# Fills each cell's clean-robust frontier more densely + wider, and traces the
# norm-adaptive cell in 2D (median temperature x dispersion). All NEW points
# (no overlap with run_iso_2x2.sh). Isolated configs already bake lamda=0,
# weight_avg=False. reform raw norm median = 6.443 (iso4 alpha = median - tau*6.443).
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

run () { # cfg tau alpha
  echo "############ $1  tau=$2 alpha=$3  START $(date) ############"
  $PY -u main.py --config_name "$1" --tau "$2" --alpha "$3"
  echo "############ $1  tau=$2 alpha=$3  DONE  $(date) ############"
}

# ---- iso3 norm const: extend frontier (have 0.4/0.5/0.6) ----
run iso3_norm_const.yaml  0  0.25
run iso3_norm_const.yaml  0  0.35
run iso3_norm_const.yaml  0  0.70
run iso3_norm_const.yaml  0  0.90

# ---- iso4 norm adapt: 2D frontier  median(m) x dispersion(tau), alpha=m-tau*6.443 ----
run iso4_norm_adapt.yaml  0.03  0.307   # m=0.5 low disp (have 0.05,0.08 at m=0.5)
run iso4_norm_adapt.yaml  0.05  0.078   # m=0.4
run iso4_norm_adapt.yaml  0.05  0.278   # m=0.6
run iso4_norm_adapt.yaml  0.08  0.085   # m=0.6 higher disp

# ---- iso1 nonorm const: extend low+high (have 5/6/7) ----
run iso1_nonorm_const.yaml  0  3.0
run iso1_nonorm_const.yaml  0  4.0
run iso1_nonorm_const.yaml  0  8.0
run iso1_nonorm_const.yaml  0  9.0

# ---- iso2 nonorm adapt: extend tau + a higher floor (have 0.75,1.0 @a=0.5) ----
run iso2_nonorm_adapt.yaml  0.5   0.5
run iso2_nonorm_adapt.yaml  1.5   0.5
run iso2_nonorm_adapt.yaml  1.0   1.0

echo "############ ALL ISO DENSE DONE $(date) ############"
