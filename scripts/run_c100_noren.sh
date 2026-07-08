#!/bin/bash
# NEW METHOD: fgsm_noren = carve (downweight FGSM-vulnerable dims) WITHOUT the mean(w)=1 renorm.
# Dropping the renorm makes mean(w)~exp(-beta), so beta ALSO softens -> one knob = shape+temperature.
# Goal: beat "/13 only" (= L2norm teacher, entropy~4.58). One-batch check: beta~3 reproduces /13 softness
#   (norm 2.09 vs 1.96, maxprob 0.020 vs 0.030) but KEEPS the carve shape (/13 has none).
# steps=10 (slow AT), carve PGD-2 (gamma=2), norm student, alpha=1 fixed (temp comes from beta now).
# Sweep softness RANGE (not just /13-match): beta=1.0~3.0 covers entropy ~4.05 -> ~4.58 (/13).
#   beta=3.0 == /13-soft (L2norm) but with input-dependent carve shape; lower beta keeps target structure.
# Results -> results/CIFAR100/fgsm_noren_s10/ (config_name based; SEPARATE from fgsm_noscale_s10).
# Baseline to beat: /13 only == L2norm-s10 teacher.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for b in 3.0 2.0 1.5 1.0; do
 echo "############ NOREN beta=$b alpha=1 START $(date) ############"
 $PY -u main.py --config_name fgsm_noren_s10.yaml --tau "$b" --gamma 2 --alpha 1.0 --dataset CIFAR100
 echo "############ NOREN beta=$b DONE $(date) ############"
done
echo "############ C100 NOREN ALL DONE $(date) ############"
