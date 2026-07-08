#!/bin/bash
# carve-ONLY (noscale) HIGH-beta: no global, no normalize, alpha=1 -> temperature must come from carve.
# small beta = too sharp; need bigger beta to soften. but beta too high corrupts class info (~0.2+).
# So sweep the soft-but-not-broken zone. steps=10, step 2/255, carve PGD-2, norm student.
# beat: global-only & L2norm = 42.13 (harmonic).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for b in 0.2 0.3 0.5; do
 echo "############ NOSCALE-HIBETA beta=$b alpha=1 START $(date) ############"
 $PY -u main.py --config_name fgsm_noscale_s10.yaml --tau "$b" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ NOSCALE-HIBETA beta=$b DONE $(date) ############"
done
echo "############ C100 NOSCALE-HIBETA ALL DONE $(date) ############"
