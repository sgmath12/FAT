#!/bin/bash
# FGSM-soft-carve: NORM student (clean+robust axis) + teacher carved by per-dim FGSM vulnerability.
# w=exp(-beta*|Phi(x)-Phi(x_fgsm)|), soft, fixed-const rescale. beta via --tau (logged).
# Usable range beta<=0.15 (verify_fgsm.py: beta>0.2 corrupts teacher class info). alpha=1.0 (raw-cell best).
# beta=0 == norm-student+raw-teacher baseline (~63.21, harmonic 41.77). BEAT iso3 harmonic 41.97?
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ # beta(=tau)
 echo "############ FGSM-CARVE beta=$1 alpha=1.0 START $(date) ############"
 FAT_TRANSFORM=fgsm_carve $PY -u main.py --config_name transform_ns.yaml --tau "$1" --alpha 1.0 --dataset CIFAR100
 echo "############ FGSM-CARVE beta=$1 alpha=1.0 DONE $(date) ############"; }
for b in 0.0 0.05 0.1 0.15 0.2; do run "$b"; done
echo "############ C100 FGSM-CARVE ALL DONE $(date) ############"
