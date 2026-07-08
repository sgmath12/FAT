#!/bin/bash
# carve-only-l2 (L2/squared fragility) tau sweep at steps=3 (fast regime). Fragile channels via PGD-2,
#   per-channel (Phi_t(x)-Phi_t(x_adv))**2, w=exp(-tau*fragility), no /13, student L2-norm. tau swept
#   INSIDE one folder: results/CIFAR100/carve_only_l2/output.log (append; parse by tau).
# tau scaled UP vs l1 (squaring shrinks |Δ|<1). steps=10 version = separate carve_only_l2_10step.yaml.
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for t in 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0 20.0; do
  $PY -u main.py --config_name carve_only_l2.yaml --tau "$t" --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 CARVE_ONLY_L2 (3step) SWEEP DONE $(date) ############"
