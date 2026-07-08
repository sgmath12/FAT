#!/bin/bash
# USER IDEA: also KD-supervise the student's CLEAN forward against the fixed teacher target
#   (currently only x_adv is KD-supervised; clean forward only updates BN stats).
#   loss += beta * KL(student(x)||target). beta=0 == baseline exactly.
#   tau16 seed0 isolated (lamda=0, weight_avg=False), 50ep. beta {0, 0.5, 1, 2, 5}.
# Results -> results/CIFAR100/temp_cleankd/output.log (parse by beta).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for b in 0 0.5 1 2 5; do
  echo ">>> temp_cleankd beta=$b $(date)"
  $PY -u main.py --config_name temp_cleankd.yaml --beta "$b" --tau 16 --seed 0 --dataset CIFAR100 >/dev/null 2>&1
done
echo "############ C100 CLEANKD DONE $(date) ############"
