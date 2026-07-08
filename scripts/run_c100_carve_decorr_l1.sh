#!/bin/bash
# decorrelated carve l1 (class_need via teacher PRED, no label leak). Tests whether removing
# vulnerable-but-class-IRRELEVANT dims -- an operation ORTHOGONAL to global temperature -- helps
# the student on its own. tau (carve strength) AND beta (class protection) both swept; low beta is
# where the diagnostic showed soft+denoise coexist. beta=0 == plain carve_only_l1 (control).
#   Results -> results/CIFAR100/carve_decorr_l1/output.log (append; parse by tau & beta). steps=3.
# Bar to clear: global temp studentNorm/teacherRaw H 41.62 ; plain carve_l1 peak H 37.10 (tau1.0).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for tau in 0.5 1.0; do
  for beta in 0 0.05 0.1 0.15 0.2; do
    $PY -u main.py --config_name carve_decorr_l1.yaml --tau "$tau" --beta "$beta" --dataset CIFAR100 >/dev/null 2>&1
  done
done
echo "############ C100 CARVE_DECORR_L1 (tau x beta) SWEEP DONE $(date) ############"
