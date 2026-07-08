#!/bin/bash
# Forward-plan #4 (MAIN BET): per-feature-dim deviation-selective distillation.
# nonorm base, isolated (lamda=0, weight_avg=False) -> directly comparable to iso1.
# beta = deviation sharpness; beta=0 == iso1 nonorm-const (alpha=13: clean 60.71/pgd 31.04/cw 25.82).
# BAR: beta>0 must BEAT iso1 on the clean-robust trade-off, not just match.
# Appends to results/CIFAR100/chansel/output.log.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

run () { # beta
  echo "############ CHANSEL alpha=13 beta=$1  START $(date) ############"
  $PY -u main.py --config_name chansel.yaml --alpha 13 --beta "$1" --dataset CIFAR100
  echo "############ CHANSEL alpha=13 beta=$1  DONE  $(date) ############"
}

for b in 0.5 1.0 2.0 4.0; do
  run "$b"
done

echo "############ C100 CHANSEL ALL DONE $(date) ############"
