#!/bin/bash
# Plan #3: distill from the SCRATCH-NORMALIZED (scale=13) teacher (clean_z).
# Teacher logits are ~13x larger (scale), so alpha~9 ≈ iso3's a0.7 operating point.
# Compare student to iso1 nonorm(60.71/31.04) and iso3 post-hoc-norm(63.49/31.37).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ echo "############ DISTILL-Z alpha=$1 START $(date) ############"
 $PY -u main.py --config_name distill_z.yaml --tau 0 --alpha "$1" --dataset CIFAR100
 echo "############ DISTILL-Z alpha=$1 DONE $(date) ############"; }
for a in 7 9 13; do run "$a"; done
echo "############ C100 DISTILL-Z ALL DONE $(date) ############"
