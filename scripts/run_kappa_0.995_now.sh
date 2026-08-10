#!/bin/bash
# kappa=0.995 (2026-07-18), run IMMEDIATELY -- 0.9999 was cancelled by user (0.9995 already came
# in worse than champion on both clean and cw, so 0.9999 was assumed to be worse still; no point
# burning GPU time confirming it). 0.995 sits between 0.9 and 0.999 (closer to 0.999), narrowing
# the bracket on the low side. Writes KAPPA_EXT3_DONE (AWP-SAM waits on this).
# Known so far: kappa 0.9 clean64.03/pgd20 31.73/cw26.80; kappa 0.999 (champion) clean62.75/
# pgd33.96/cw28.41; kappa 0.9995 clean57.06/pgd31.84/cw26.62; kappa 0.9999 cancelled (unrun).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/kappa_sweep_chain.log
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

k=0.995
echo "=== kappa=${k} SMOKE START $(date) ===" >> $LOG
if $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --kappa $k --epochs 1 \
    > results/CIFAR100/kappa_${k}_smoke_driver.log 2>&1; then
  echo "=== kappa=${k} smoke OK, full START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --kappa $k \
    > results/CIFAR100/kappa_${k}_driver.log 2>&1
  cp $CKDIR/feat_direction_last.pkl $CKDIR/kappa_${k}_last.pkl 2>/dev/null
  echo "=== kappa=${k} DONE (ckpt archived) $(date) ===" >> $LOG
else
  echo "!!! kappa=${k} SMOKE FAILED, skipped $(date) ===" >> $LOG
fi

echo "KAPPA_EXT3_DONE $(date)" >> $LOG
