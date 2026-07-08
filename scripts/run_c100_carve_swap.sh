#!/bin/bash
# CARVE+SWAP sweep: decorr carve + global temp + swap-rectification. beta fixed 0.1 (saturated).
#   (tau0.5,T16) first = matched point vs non-swap carve 41.79. Then tau x T density.
#   Results -> results/CIFAR100/carve_decorr_temp_swap_l1/output.log (parse by tau,beta,temperature). steps=3.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ $PY -u main.py --config_name carve_decorr_temp_swap_l1.yaml --tau "$1" --beta 0.1 --temperature "$2" --dataset CIFAR100 >/dev/null 2>&1; }
for T in 16 12 24 8; do run 0.5 "$T"; done
for T in 16 12 24 8; do run 1.0 "$T"; done
echo "############ C100 CARVE_SWAP sweep DONE $(date) ############"
