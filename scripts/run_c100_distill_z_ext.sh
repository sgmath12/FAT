#!/bin/bash
# Extend distill_z alpha sweep — clean was still RISING at alpha=13 (eff temp 1.0),
# iso3's good region is eff temp ~0.7-1.6, so map alpha 16/20/26 (eff 1.2/1.5/2.0) to find the peak.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ echo "############ DISTILL-Z-EXT alpha=$1 START $(date) ############"
 $PY -u main.py --config_name distill_z.yaml --tau 0 --alpha "$1" --dataset CIFAR100
 echo "############ DISTILL-Z-EXT alpha=$1 DONE $(date) ############"; }
for a in 16 20 26; do run "$a"; done
echo "############ C100 DISTILL-Z-EXT ALL DONE $(date) ############"
