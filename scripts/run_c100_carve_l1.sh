#!/bin/bash
# carve-only-l1 (L1 fragility) FULL tau sweep at steps=3 (fast regime). Fragile channels via PGD-2,
#   per-channel |Phi_t(x)-Phi_t(x_adv)|, w=exp(-tau*fragility), no /13, student L2-norm. tau swept
#   INSIDE one folder: results/CIFAR100/carve_only_l1/output.log (append; parse by tau).
# NOTE: steps=10 version is run separately later via carve_only_l1_10step.yaml.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for t in 0.01 0.05 0.1 0.15 0.2 0.3 0.5 1.0 2.0; do
  $PY -u main.py --config_name carve_only_l1.yaml --tau "$t" --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 CARVE_ONLY_L1 (3step) SWEEP DONE $(date) ############"
