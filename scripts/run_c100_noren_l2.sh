#!/bin/bash
# CARVE-ONLY noren with L2 (squared) per-dim vulnerability: dev = (Phi_clean - Phi_adv)**2 (vs L1 abs).
# Squared concentrates the carve on the few most-vulnerable dims, gentler elsewhere -> may keep structure
#   better at matched softness than L1 (which monotonically HURT: best L1 beta=0.25 -> harmonic 38.62 < 42.13).
# alpha=1 FIXED, NO global, NO renorm. transform=fgsm_noren_l2. steps=10 (2/255), carve PGD-2 (gamma=2).
# Ascending beta: 0.0 (= raw teacher anchor, carve_w=1, identical for L1/L2) first, then increasing carve.
# Results APPEND -> results/CIFAR100/fgsm_noren_l2_s10/output.log (parse by tau; harmonic=2cp/(c+p)).
# Baseline to beat: /13 == L2norm = clean 62.6 / pgd20 31.7 / HARMONIC 42.13.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for b in 0.0 0.25 0.5 0.75 1.0 1.5 2.0 3.0; do
 echo "############ NOREN-L2 beta=$b alpha=1 START $(date) ############"
 $PY -u main.py --config_name fgsm_noren_l2_s10.yaml --tau "$b" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ NOREN-L2 beta=$b DONE $(date) ############"
done
echo "############ C100 NOREN-L2 ALL DONE $(date) ############"
