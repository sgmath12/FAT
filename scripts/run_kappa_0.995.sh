#!/bin/bash
# kappa=0.995 (2026-07-18, user q): 0.999 is looking like a local sweet spot -- both neighbors
# tested so far are worse (0.9 -> cw26.80, 0.9995 -> cw26.62, both below champion's cw28.41).
# 0.995 sits BETWEEN 0.9 and 0.999 (closer to 0.999) -- narrows the bracket on the low side to
# see how sharp the peak is. Chained AFTER kappa=0.9999 (waits for KAPPA_EXT_DONE), writes a
# new marker (KAPPA_EXT2_DONE) so AWP-SAM (re-pointed to wait for this) still runs last.
# Known so far: kappa 0.9 clean64.03/pgd20 31.73/cw26.80; kappa 0.999 (champion) clean62.75/
# pgd33.96/cw28.41; kappa 0.9995 clean57.06/pgd31.84/cw26.62; kappa 0.9999 in flight.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/kappa_sweep_chain.log
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

until grep -q "KAPPA_EXT_DONE" $LOG 2>/dev/null; do sleep 180; done

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

echo "KAPPA_EXT2_DONE $(date)" >> $LOG
