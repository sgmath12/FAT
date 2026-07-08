#!/bin/bash
# WA + consistency(lamda) sweep on the two best baselines (tau=16, seed0).
#   configs: temp_studentNorm_teacherRaw_wa (method=temperature)
#            temp_studentNorm_teacherRaw_swap_wa (method=temperature_swap)
#   both weight_avg=True. lamda in {0,2,5,10} (0 = WA-only, isolates WA gain).
#   Results -> results/CIFAR100/<config>/output.log (append; parse by lamda in Experiment Configuration line).
set -u; cd /mnt/d/research/FAT; export PYTHONPATH=/mnt/d/research/FAT
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
for cfg in temp_studentNorm_teacherRaw_wa temp_studentNorm_teacherRaw_swap_wa; do
  for lam in 0 2 5 10; do
    echo ">>> $cfg lamda=$lam tau=16 seed=0 $(date)"
    $PY -u main.py --config_name ${cfg}.yaml --tau 16 --lamda "$lam" --seed 0 --dataset CIFAR100 >/dev/null 2>&1
  done
done
echo "############ C100 WA+LAMDA sweep DONE $(date) ############"
