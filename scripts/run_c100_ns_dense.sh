#!/bin/bash
# transform_ns DENSE: norm-student + RAW teacher, extend alpha (had 0.5/0.7/1.0, best=1.0/41.77, not peaked).
# alpha via --alpha; transform=raw via env. Find the clean-robust peak of the student-norm baseline.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ echo "############ NS-DENSE teacher=raw alpha=$1 START $(date) ############"
 FAT_TRANSFORM=raw $PY -u main.py --config_name transform_ns.yaml --alpha "$1" --dataset CIFAR100
 echo "############ NS-DENSE teacher=raw alpha=$1 DONE $(date) ############"; }
for a in 1.3 1.6 2.0; do run "$a"; done
echo "############ C100 NS-DENSE ALL DONE $(date) ############"
