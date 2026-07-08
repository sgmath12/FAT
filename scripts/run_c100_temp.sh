#!/bin/bash
# Temperature study (FAST, steps=3): 4 norm-combo cells x tau sweep.
#   student_norm{Norm,Raw} x teacher_norm{Norm,Raw} = 4 configs (one folder each);
#   tau swept INSIDE each folder (append to that cell's output.log, parse by tau).
# method=temperature: target = teacher_logits / tau (tau=1 == raw teacher when teacher_norm=Raw).
# Results ONLY in results/CIFAR100/<config>/output.log (self-labels student_norm/teacher_norm/tau).
# No console logs saved (stdout -> /dev/null). Report clean AND pgd20; harmonic = 2*clean*pgd20/(clean+pgd20).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
TAUS="1 4"
for cfg in temp_studentNorm_teacherRaw temp_studentNorm_teacherNorm temp_studentRaw_teacherRaw temp_studentRaw_teacherNorm; do
  for t in $TAUS; do
    $PY -u main.py --config_name ${cfg}.yaml --tau "$t" --dataset CIFAR100 >/dev/null 2>&1
  done
done
