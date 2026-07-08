#!/bin/bash
# B PASS1 REMAINING cells only (append to existing results/CIFAR100/carve_decorr_temp_l1/output.log).
# Target grid beta{0,0.1,0.2} x T{8,16,24,32}; already done: beta0/T16, beta0/T24. tau=0.5, steps=3.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ $PY -u main.py --config_name carve_decorr_temp_l1.yaml --tau 0.5 --beta "$1" --temperature "$2" --dataset CIFAR100 >/dev/null 2>&1; }
# decorr cells first (the actual test)
for T in 16 24 32 8; do run 0.1 "$T"; done
for T in 16 24 32 8; do run 0.2 "$T"; done
# finish beta0 control column
run 0 32
run 0 8
echo "############ C100 CARVE_DECORR_TEMP_L1 PASS1 REMAINING DONE $(date) ############"
