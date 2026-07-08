#!/bin/bash
# powernorm RE-RUN (BN-fixed: now does clean forward like iso). p via --tau (logged as tau).
# Order: p=1 FIRST (validate it now ~= iso3 63.5), then p>1 (1.25,1.5,2 = over-suppress overconfident,
# can it BEAT iso3?), then smaller (0.75,0.5,0.25,0=raw). transform=powernorm set in yaml.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ # p(=tau)
 echo "############ POWERNORM2 p=$1 alpha=0.7 START $(date) ############"
 $PY -u main.py --config_name transform.yaml --tau "$1" --alpha 0.7 --dataset CIFAR100
 echo "############ POWERNORM2 p=$1 alpha=0.7 DONE $(date) ############"; }
for p in 1.0 1.25 1.5 2.0 0.75 0.5 0.25 0.0; do run "$p"; done
echo "############ C100 TRANSFORM2 ALL DONE $(date) ############"
