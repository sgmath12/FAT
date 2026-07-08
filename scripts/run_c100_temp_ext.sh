#!/bin/bash
# Extension of run_c100_temp.sh: higher-tau tail (appends to same per-config output.log).
#   all cells get tau 8, 16 EXCEPT studentNorm_teacherNorm (tau 8 only).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
run(){ $PY -u main.py --config_name ${1}.yaml --tau "$2" --dataset CIFAR100 >/dev/null 2>&1; }
for cfg in temp_studentNorm_teacherRaw temp_studentRaw_teacherRaw temp_studentRaw_teacherNorm; do
  for t in 8 16; do run "$cfg" "$t"; done
done
run temp_studentNorm_teacherNorm 8
echo "############ C100 TEMP EXT ALL DONE $(date) ############"
