#!/bin/bash
# Softer-than-champion tau extension (user, 2026-07-17), behind NORMFEAT_TARGET_DONE.
# Grid so far {1,2,4,8} (all sharper than champion's tau16) all LOST, pgd crashed hard while
# cw stayed flat. This checks the OTHER direction: is tau16 the true peak, or does even softer
# (tau24,32) help further? beta=1 (champion default) for a clean one-axis read.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa
until grep -q "NORMFEAT_TARGET_DONE" $LOG 2>/dev/null; do sleep 180; done
for t in 24 32; do
  echo "=== tauhigh tau=${t} beta=1 START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --tau $t --beta 1.0 \
    > results/CIFAR100/tauhigh_t${t}_driver.log 2>&1
  cp $CKDIR/feat_direction_last.pkl $CKDIR/tauhigh_t${t}_last.pkl 2>/dev/null
  echo "=== tauhigh tau=${t} beta=1 DONE (ckpt archived) $(date) ===" >> $LOG
done
echo "TAUHIGH_DONE $(date)" >> $LOG
