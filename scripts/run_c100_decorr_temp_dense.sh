#!/bin/bash
# DENSE beta x tau search for decorr+temp (find a bigger gap than the +0.17 at beta0.1/T16=41.79).
# T fixed at 16 (Pass1 best; T24/32 worse). Plus a low-T probe at the current best. steps=3.
# Appends to results/CIFAR100/carve_decorr_temp_l1/output.log (parse by tau,beta,temperature).
# Already have: (beta0.1,tau0.5,T16)=41.79, (beta0,*)=loses. Baseline 41.62.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ $PY -u main.py --config_name carve_decorr_temp_l1.yaml --tau "$1" --beta "$2" --temperature "$3" --dataset CIFAR100 >/dev/null 2>&1; }
# 1) nail T at the current best (T16>T24 -> lower T may win)
run 0.5 0.1 12
run 0.5 0.1 8
# 2) finer beta at the known-good tau
run 0.5 0.05 16
run 0.5 0.15 16
# 3) tau density x beta (T16); skip (tau0.5,beta0.1)=done
for tau in 0.3 0.7 1.0; do
  for beta in 0.05 0.1 0.15; do
    run "$tau" "$beta" 16
  done
done
echo "############ C100 DECORR_TEMP DENSE (betaxtau @T16 + Tprobe) DONE $(date) ############"
