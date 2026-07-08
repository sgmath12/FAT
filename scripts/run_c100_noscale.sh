#!/bin/bash
# STEP1: carve ONLY (no global scale, alpha=1 fixed) — does carve alone soften the target & help?
# + L2norm-teacher baseline (steps=10). norm student, slow AT steps=10, carve PGD-2.
# Compare harmonic vs: global-only (fgsm_s10 beta=0 = 42.23) and L2norm-teacher.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
# (a) L2norm teacher baseline (steps=10), alpha sweep small
echo "############ L2NORM-S10 alpha=0.7 START $(date) ############"
$PY -u main.py --config_name l2norm_s10.yaml --alpha 0.7 --dataset CIFAR100
echo "############ L2NORM-S10 alpha=0.7 DONE $(date) ############"
# (b) carve only (noscale), alpha=1 fixed, beta sweep
for b in 0.0 0.05 0.1 0.15 0.2; do
 echo "############ NOSCALE beta=$b alpha=1 START $(date) ############"
 $PY -u main.py --config_name fgsm_noscale.yaml --tau "$b" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ NOSCALE beta=$b alpha=1 DONE $(date) ############"
done
echo "############ C100 NOSCALE ALL DONE $(date) ############"
