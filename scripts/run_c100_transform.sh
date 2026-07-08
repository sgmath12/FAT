#!/bin/bash
# PRINCIPLED post-hoc carve: powernorm dose-response Phi/||Phi||^p.
# Mechanism: ||Phi|| encodes non-robust overconfidence (corr(norm,margin)=0.59). p = how much to strip.
# p=0 == raw(iso1 ~60.7), p=1 == L2(iso3 ~63.5). Find the optimal dose; p>1 = over-suppress overconfident.
# (standardize/whiten dropped: no robustness rationale, per discussion.)
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ # p alpha
 echo "############ POWERNORM p=$1 alpha=$2 START $(date) ############"
 FAT_TRANSFORM=powernorm FAT_PNORM="$1" $PY -u main.py --config_name transform.yaml --alpha "$2" --dataset CIFAR100
 echo "############ POWERNORM p=$1 alpha=$2 DONE $(date) ############"; }
# dose-response at alpha=0.7 (iso3's best operating temp); p=0 raw, p=1 L2
for p in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 2.0; do run "$p" 0.7; done
echo "############ C100 TRANSFORM ALL DONE $(date) ############"
