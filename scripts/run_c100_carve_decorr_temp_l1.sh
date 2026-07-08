#!/bin/bash
# B decoupled (decorr carve + global temperature), PASS 1: tau=0.5 fixed, beta x T grid.
#   target = teacher.linear(Phi_t*w)/T. beta {0.1,0.2}=decorr+temp, beta 0=carve+temp control.
#   T swept incl. low(8) and high(24,32) -- decorr re-sharpens so optimum T may sit high. steps=3.
#   Results -> results/CIFAR100/carve_decorr_temp_l1/output.log (parse by beta & temperature).
# Bar: global temp H 41.62. Reference: carve+temp beta0/T16 = H 40.09 (clean 63.35). Pass 2: sweep tau.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for beta in 0.1 0.2 0; do
  for T in 16 24 32 8; do
    $PY -u main.py --config_name carve_decorr_temp_l1.yaml --tau 0.5 --beta "$beta" --temperature "$T" --dataset CIFAR100 >/dev/null 2>&1
  done
done
echo "############ C100 CARVE_DECORR_TEMP_L1 PASS1 (tau0.5, beta{0.1,0.2,0} x T{8,16,24,32}) DONE $(date) ############"
