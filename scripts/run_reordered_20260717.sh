#!/bin/bash
# Reordered per user (2026-07-17 09:00): normfeat_target + tauhigh(24,32) FIRST, then resume
# the remaining taubeta grid cells (tau4/beta4 restarted since cancelled mid-run; tau8/beta0.5;
# tau8/beta2). Champion bar: k350+WA+lamda4 tau16/beta1 = clean62.75/pgd33.96/cw28.41/AA26.29.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

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

for t in 24 32; do
  echo "=== tauhigh tau=${t} beta=1 START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --tau $t --beta 1.0 \
    > results/CIFAR100/tauhigh_t${t}_driver.log 2>&1
  cp $CKDIR/feat_direction_last.pkl $CKDIR/tauhigh_t${t}_last.pkl 2>/dev/null
  echo "=== tauhigh tau=${t} beta=1 DONE (ckpt archived) $(date) ===" >> $LOG
done
echo "TAUHIGH_DONE $(date)" >> $LOG

TAUBETA=( "4 4" "8 0.5" "8 2" )
for tb in "${TAUBETA[@]}"; do
  set -- $tb; t=$1; b=$2
  echo "=== taubeta tau=${t} beta=${b} START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --tau $t --beta $b \
    > results/CIFAR100/taubeta_t${t}_b${b}_driver.log 2>&1
  cp $CKDIR/feat_direction_last.pkl $CKDIR/taubeta_t${t}_b${b}_last.pkl 2>/dev/null
  echo "=== taubeta tau=${t} beta=${b} DONE (ckpt archived) $(date) ===" >> $LOG
done
echo "TAUBETA_SWEEP_DONE $(date)" >> $LOG
