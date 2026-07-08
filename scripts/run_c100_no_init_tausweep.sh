#!/bin/bash
# no_init ablation of the temperature-study winner cell (studentNorm/teacherRaw, FAST steps=3):
# student starts from RANDOM init (finetune=False) instead of the clean checkpoint; teacher unchanged.
# tau swept to match the original temp_studentNorm_teacherRaw sweep for direct comparison.
# Results ONLY in results/CIFAR100/temp_studentNorm_teacherRaw_no_init/output.log (self-labels tau).
# No console logs saved (stdout -> /dev/null). Report clean AND pgd20; harmonic = 2*clean*pgd20/(clean+pgd20).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
TAUS="1 4 8 10 12 16 20 24"
for t in $TAUS; do
  $PY -u main.py --config_name temp_studentNorm_teacherRaw_no_init.yaml --tau "$t" --dataset CIFAR100 >/dev/null 2>&1
done
