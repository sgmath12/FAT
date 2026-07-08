#!/bin/bash
# margin + reformation=True, FAIR (centered) sweep.
#
# Earlier alpha=0.5 runs were all "too hot": T = tau*margin + alpha with alpha=0.5
# forces median T > 1.0, well above the known sweet spot. const_reform (tau=0,
# pure L2-norm + constant T=0.5) already hits pgd20=54.90 / cw=52.57 == the
# DPFAT_adaptive baseline (54.88 / 52.51). So the adaptive kappa(x) added NOTHING.
#
# This sweep gives margin-adaptivity its fairest shot: hold MEDIAN temperature at
# the sweet spot 0.5 and let tau inject dispersion AROUND it.
#   reform margin (top1-top2): mean 1.19, p5 0.25, p50 1.33, p95 1.62
#   alpha = 0.5 - tau*p50  =>  T(median)=0.5 for every run, dispersion grows with tau.
#
# If even centered margin-adaptivity is <= const_reform (54.90), per-sample
# temperature is confirmed inert under correct normalization -- closes the cell.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

# (tau, alpha) with median T fixed at 0.5; dispersion spread in comment
RUNS=(
  "0.1 0.367"   # spread 1.35x  (mild)
  "0.2 0.234"   # spread 1.97x
  "0.3 0.101"   # spread 3.34x  (strong; T(p5)=0.18 sharp on ambiguous samples)
)

for pair in "${RUNS[@]}"; do
  set -- $pair
  tau=$1; alpha=$2
  echo "############ margin_reform CENTERED tau=$tau alpha=$alpha START $(date) ############"
  $PY -u main.py --config_name DPFAT_margin_reform.yaml --tau "$tau" --alpha "$alpha"
  echo "############ margin_reform CENTERED tau=$tau alpha=$alpha DONE  $(date) ############"
done

echo "############ ALL DONE $(date) ############"
