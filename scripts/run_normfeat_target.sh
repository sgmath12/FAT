#!/bin/bash
# Normalized-teacher-feature head target, behind TAUBETA_SWEEP_DONE (2026-07-17).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py
until grep -q "TAUBETA_SWEEP_DONE" $LOG 2>/dev/null; do sleep 180; done
echo "=== normfeat_target SMOKE START $(date) ===" >> $LOG
if $PY -u main.py --config_name featdir_normfeat_target.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --epochs 1 \
    > results/CIFAR100/normfeat_target_smoke_driver.log 2>&1; then
  echo "=== normfeat_target smoke OK, full START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_normfeat_target.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/normfeat_target_driver.log 2>&1
  echo "=== normfeat_target DONE $(date) ===" >> $LOG
  $PY -u $EVAL "k350+WA+lamda4+normfeat_target|CIFAR100/checkpoint/featdir_normfeat_target/feat_direction_last.pkl" >> results/CIFAR100/aa_verify_20260716.log 2>&1
else
  echo "!!! normfeat_target SMOKE FAILED $(date)" >> $LOG
fi
echo "NORMFEAT_TARGET_DONE $(date)" >> $LOG
