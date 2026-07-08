#!/bin/bash
# ISOLATED 2x2: does the softening MECHANISM matter once consistency(lamda) + SWA are OFF?
# All runs: lamda=0, weight_avg=False (baked into the iso*.yaml configs) -> KL term only.
# Question per row: does ADAPTIVE (tau>0) beat CONST (tau=0) at a matched operating point?
# Each cell swept over its temperature knob (fair frontier, not a single point).
#
#                 | const (tau=0)            | adaptive (tau>0)
#   no-norm (F)   | iso1: alpha {6,5,7}      | iso2: tau {1.0,0.75} a=0.5
#   norm    (T)   | iso3: alpha {0.5,0.4,0.6}| iso4: tau {0.05,0.08} centered
#
# reform raw feat norm: mean 6.456, p5 5.28, p95 7.65 (disp 1.45x). iso4 centered:
#   alpha = 0.5 - tau*6.443 to hold median kappa ~0.5; tau=0.05->a=0.18, tau=0.08->a=0.0.
# DECISIVE norm row runs FIRST (gives the (A) all-flat vs (B) adaptive-works verdict early).
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

run () { # cfg tau alpha
  echo "############ $1  tau=$2 alpha=$3  START $(date) ############"
  $PY -u main.py --config_name "$1" --tau "$2" --alpha "$3"
  echo "############ $1  tau=$2 alpha=$3  DONE  $(date) ############"
}

# ---- NORM row (decisive) ----
run iso3_norm_const.yaml  0     0.5
run iso3_norm_const.yaml  0     0.4
run iso3_norm_const.yaml  0     0.6
run iso4_norm_adapt.yaml  0.05  0.18
run iso4_norm_adapt.yaml  0.08  0.0

# ---- NO-NORM row ----
run iso1_nonorm_const.yaml  0     6.0
run iso1_nonorm_const.yaml  0     5.0
run iso1_nonorm_const.yaml  0     7.0
run iso2_nonorm_adapt.yaml  1.0   0.5
run iso2_nonorm_adapt.yaml  0.75  0.5

echo "############ ALL ISO 2x2 DONE $(date) ############"
