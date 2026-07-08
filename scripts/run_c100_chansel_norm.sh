#!/bin/bash
# #4 on NORM base: per-dim DIRECTIONAL deviation selection on top of L2-normalization.
# Tests paper §4 thesis (non-robustness in directions after magnitude removed by norm).
# beta=0 == iso3 norm-const baseline (a0.7: 63.49/31.37/26.84); beta>0 must BEAT it.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run () {
  echo "############ CHANSEL-NORM alpha=0.7 beta=$1  START $(date) ############"
  $PY -u main.py --config_name chansel_norm.yaml --alpha 0.7 --beta "$1" --dataset CIFAR100
  echo "############ CHANSEL-NORM alpha=0.7 beta=$1  DONE  $(date) ############"
}
for b in 0.5 1.0 2.0 4.0; do run "$b"; done
echo "############ C100 CHANSEL-NORM ALL DONE $(date) ############"
