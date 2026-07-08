#!/bin/bash
# fgsm soft DENSE (around sweet spot 0.1) + fgsm HARD mask (top-k vulnerable dims). norm student, alpha=1.0.
# baseline harmonic 41.77 (norm-student + raw teacher). beat it = a non-normalization teacher carve works.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
soft(){ echo "############ FGSM2 soft beta=$1 START $(date) ############"
 FAT_TRANSFORM=fgsm_carve $PY -u main.py --config_name transform_ns.yaml --tau "$1" --alpha 1.0 --dataset CIFAR100
 echo "############ FGSM2 soft beta=$1 DONE $(date) ############"; }
hard(){ echo "############ FGSM2 hard k=$1 fill=$2 START $(date) ############"
 FAT_TRANSFORM=fgsm_hard $PY -u main.py --config_name transform_ns.yaml --tau "$1" --pct "$2" --alpha 1.0 --dataset CIFAR100
 echo "############ FGSM2 hard k=$1 fill=$2 DONE $(date) ############"; }
# soft dense near sweet spot
soft 0.025; soft 0.075; soft 0.125
# hard mask: top-k vulnerable dims -> 0 (k<=5 safe), + soft-hard fill=0.1
hard 3 0.0; hard 5 0.0; hard 10 0.0; hard 5 0.1; hard 10 0.1
echo "############ C100 FGSM2 ALL DONE $(date) ############"
