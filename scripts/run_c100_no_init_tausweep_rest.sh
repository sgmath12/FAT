#!/bin/bash
# Resume of run_c100_no_init_tausweep.sh: taus 1/4/8/10 already completed (see output.log);
# tau=12 was killed mid-run on 2026-07-07 -> rerun 12 and the remaining 16/20/24.
# Results append to results/CIFAR100/temp_studentNorm_teacherRaw_no_init/output.log (self-labels tau).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
TAUS="12 16 20 24"
for t in $TAUS; do
  $PY -u main.py --config_name temp_studentNorm_teacherRaw_no_init.yaml --tau "$t" --dataset CIFAR100 >/dev/null 2>&1
done
