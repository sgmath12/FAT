#!/bin/bash
# Longer-training check for the BEST margin+rank config (tau=0.5, alpha=0.5;
# best pgd20 53.87 in the 50-epoch sweep). Question: does margin+rank close the
# gap to the DPFAT_adaptive baseline (pgd20 54.88) with more epochs?
#
# NOTE this is NOT just "more of the same": main.py/methods.py rescale the cyclic
# LR schedule, annealing=(epoch/epochs)**2, and weight-avg decay to config.epochs,
# so 100/200-epoch runs have genuinely different dynamics.
#
# Appends to results/CIFAR10/DPFAT_rank/output.log (Experiment Configuration header
# carries epochs=100/200 to distinguish). Checkpoints overwrite each run.
set -u
cd /mnt/d/research/FAT
export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python

for ep in 100 200; do
  echo "############ margin+rank BEST  tau=0.5 alpha=0.5 epochs=$ep  START $(date) ############"
  $PY -u main.py --config_name DPFAT_rank.yaml --tau 0.5 --alpha 0.5 --epochs "$ep"
  echo "############ margin+rank BEST  tau=0.5 alpha=0.5 epochs=$ep  DONE  $(date) ############"
done

echo "############ LONG ALL DONE $(date) ############"
