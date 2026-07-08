#!/bin/bash
# Finish Pass1 grid beta{0,0.1,0.2} x T{8,16,24,32} @ tau0.5. Done: (0,16)(0,24)(0.1,16)(0.1,24).
# beta 0.2 first (what was next), then leftover T8/T32. Appends to carve_decorr_temp_l1/output.log. steps=3.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ $PY -u main.py --config_name carve_decorr_temp_l1.yaml --tau 0.5 --beta "$1" --temperature "$2" --dataset CIFAR100 >/dev/null 2>&1; }
for T in 16 24 32 8; do run 0.2 "$T"; done
run 0.1 32; run 0.1 8
run 0   32; run 0   8
echo "############ C100 DECORR_TEMP PASS1 FULL DONE $(date) ############"
