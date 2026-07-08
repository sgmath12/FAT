#!/bin/bash
# SWAP experiment: key 2x2 comparison points FIRST, then density.
#   baseline+swap (temp_swap) and carve+swap at their matched hyperparams, so the core question
#   ("does swap help; does carve still win under swap") answers fast; density fills after.
# Results -> results/CIFAR100/temp_studentNorm_teacherRaw_swap/  and  .../carve_decorr_temp_swap_l1/
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
tmp(){ $PY -u main.py --config_name temp_studentNorm_teacherRaw_swap.yaml --tau "$1" --dataset CIFAR100 >/dev/null 2>&1; }
crv(){ $PY -u main.py --config_name carve_decorr_temp_swap_l1.yaml --tau "$1" --beta 0.1 --temperature "$2" --dataset CIFAR100 >/dev/null 2>&1; }
# --- KEY 2x2 points first ---
tmp 16          # baseline+swap  (vs non-swap 41.62)
crv 0.5 16      # carve+swap     (vs non-swap 41.79)
echo "############ SWAP key 2x2 points DONE $(date) ############"
# --- density ---
for tau in 12 24 8; do tmp "$tau"; done
for T in 12 24 8; do crv 0.5 "$T"; done
for T in 16 12 24 8; do crv 1.0 "$T"; done
echo "############ C100 SWAP_ALL DONE $(date) ############"
