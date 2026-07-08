#!/bin/bash
# Chained behind the no_init tau sweep (2026-07-07): waits for the running
# temp_studentNorm_teacherRaw_no_init runs (scripts/run_c100_no_init_tausweep_rest.sh) to finish,
# then runs the per-class-temperature bilevel experiment (temp_tauclass_bilevel.yaml).
# Fair bar: temp_baseline_val45k (same 45000-image split). Watch tauc_std in
# results/CIFAR100/temp_tauclass_bilevel/output.log.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
while pgrep -f "no_init_tausweep_rest|temp_studentNorm_teacherRaw_no_init.yaml" > /dev/null; do sleep 60; done
echo "no_init sweep done, starting tauclass_bilevel $(date)"
$PY -u main.py --config_name temp_tauclass_bilevel.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/tauclass_bilevel_driver.log 2>&1
echo "TAUCLASS_BILEVEL_DONE $(date)"
