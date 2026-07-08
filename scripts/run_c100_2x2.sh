#!/bin/bash
# Fill the 2x2 normalization decomposition (norm-student row) + validate iso3 in unified codepath.
#   FAT_TRANSFORM=raw    + reformation=True  => NORM student + RAW teacher = THE MISSING CELL
#   FAT_TRANSFORM=l2norm + reformation=True  => NORM student + NORM teacher = should reproduce iso3 (~63.5)
# Tests: (1) does student-norm alone give the clean boost? (2) user's hypothesis — is iso3 good because
# BOTH normalized = matched feature space? (norm-student + raw-teacher = MISMATCHED).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ # transform alpha
 echo "############ 2X2 student=norm teacher=$1 alpha=$2 START $(date) ############"
 FAT_TRANSFORM="$1" $PY -u main.py --config_name transform_ns.yaml --alpha "$2" --dataset CIFAR100
 echo "############ 2X2 student=norm teacher=$1 alpha=$2 DONE $(date) ############"; }
# MISSING CELL: norm student + raw teacher (alpha frontier)
run raw 0.5
run raw 0.7
run raw 1.0
# VALIDATION: norm student + norm teacher == iso3 (~63.5)?
run l2norm 0.7
echo "############ C100 2X2 ALL DONE $(date) ############"
