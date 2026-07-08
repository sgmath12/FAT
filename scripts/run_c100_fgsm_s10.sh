#!/bin/bash
# SLOW AT (PGD steps=10) + carve via PGD-2 (gamma=2). norm student, alpha=1.0.
# beta dense near 0. beat beta=0 (no carve) = a non-norm teacher carve helps under heavy AT.
# separate config fgsm_s10.yaml -> results/CIFAR100/fgsm_s10/ (no folder mixing).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ echo "############ FGSM_S10 beta=$1 (AT steps=10, carve PGD-2) START $(date) ############"
 FAT_TRANSFORM=fgsm_carve $PY -u main.py --config_name fgsm_s10.yaml --tau "$1" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ FGSM_S10 beta=$1 DONE $(date) ############"; }
for b in 0.0 0.01 0.025 0.05; do run "$b"; done
echo "############ C100 FGSM_S10 ALL DONE $(date) ############"
