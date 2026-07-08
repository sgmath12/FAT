#!/bin/bash
# steps=10, step_size 2/255 (AT), carve PGD-2 (gamma=2, step eps/2=4/255). norm student.
# Clean comparison of TEACHER processing on harmonic mean — all in the SAME regime:
#   global-only (β=0) | L2norm | carve+global (β) | carve-only/noscale (β, alpha=1)
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
# 1) baselines
echo "############ S10CMP global-only (fgsm_glob beta=0) START $(date) ############"
$PY -u main.py --config_name fgsm_glob_s10.yaml --tau 0 --gamma 2 --alpha 1.0 --dataset CIFAR100
echo "############ S10CMP global-only DONE $(date) ############"
echo "############ S10CMP L2norm-teacher START $(date) ############"
$PY -u main.py --config_name l2norm_s10.yaml --alpha 0.7 --dataset CIFAR100
echo "############ S10CMP L2norm-teacher DONE $(date) ############"
# 2) carve WITH global (re-confirm @2/255)
for b in 0.025 0.05; do
 echo "############ S10CMP carve+global beta=$b START $(date) ############"
 $PY -u main.py --config_name fgsm_glob_s10.yaml --tau "$b" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ S10CMP carve+global beta=$b DONE $(date) ############"; done
# 3) STEP1: carve ONLY (noscale), alpha=1 fixed, beta sweep
for b in 0.0 0.05 0.1 0.15; do
 echo "############ S10CMP noscale beta=$b START $(date) ############"
 $PY -u main.py --config_name fgsm_noscale_s10.yaml --tau "$b" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ S10CMP noscale beta=$b DONE $(date) ############"; done
echo "############ C100 S10CMP ALL DONE $(date) ############"
