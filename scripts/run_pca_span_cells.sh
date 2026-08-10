#!/bin/bash
# Informed-subspace cells v2 (2026-07-14, behind SPAN_K_CURVE_DONE; scratch-robust reference
# REPLACED after user caught the basis mismatch -- npz 'robust' now = at_ce_freehead clean-init AT
# (participation 48.7 / r90 247), 'kdstudent' = baseline robust student (77.6 / 177) = oracle.
# NOTE the rank-collapse claim (8.7) was a from-scratch artifact; aligned robust models SPREAD rank.
# All at k=50 (< natural r90=75, where selection content can matter) vs the k-curve's random-k50.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "SPAN_K_CURVE_DONE" $LOG 2>/dev/null; do sleep 120; done
run () { echo "=== $1 eta=$2 START $(date) ===" >> $LOG; \
  $PY -u main.py --config_name $1.yaml --dataset CIFAR100 --seed 0 --eta $2 \
    > results/CIFAR100/$1_k$2_driver.log 2>&1; \
  echo "=== $1 eta=$2 DONE $(date) ===" >> $LOG; }
echo "=== pca smoke START $(date) ===" >> $LOG
if $PY -u main.py --config_name featdir_span_pcarobust.yaml --dataset CIFAR100 --seed 0 --eta 50 --epochs 1 \
    > results/CIFAR100/pca_smoke_driver.log 2>&1; then
  echo "=== pca smoke OK $(date) ===" >> $LOG
  run featdir_span_pcarobust 50
  run featdir_span_pcanatural 50
  run featdir_span_pcakdstudent 50
else
  echo "!!! pca smoke FAILED $(date)" >> $LOG
fi
echo "PCA_SPAN_CELLS_DONE $(date)" >> $LOG
