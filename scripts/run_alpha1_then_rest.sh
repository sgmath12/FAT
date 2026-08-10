#!/bin/bash
# (2026-07-17) waits for the IN-FLIGHT normfeat_target run, does its AA, then alpha=1 cell
# (prioritized ahead of tauhigh/remaining taubeta per user), then resumes the rest.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

while ps aux | grep -q "[m]ain.py --config_name featdir_normfeat_target"; do sleep 60; done
echo "=== normfeat_target DONE $(date) ===" >> $LOG
$PY -u $EVAL "k350+WA+lamda4+normfeat_target|CIFAR100/checkpoint/featdir_normfeat_target/feat_direction_last.pkl" >> results/CIFAR100/aa_verify_20260716.log 2>&1
echo "NORMFEAT_TARGET_DONE $(date)" >> $LOG

echo "=== alpha1 champion START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_alpha1_champion.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
  > results/CIFAR100/alpha1_champion_driver.log 2>&1
echo "=== alpha1 champion DONE $(date) ===" >> $LOG
$PY -u $EVAL "k350+WA+lamda4+alpha1|CIFAR100/checkpoint/featdir_alpha1_champion/feat_direction_last.pkl" >> results/CIFAR100/aa_verify_20260716.log 2>&1
echo "ALPHA1_DONE $(date)" >> $LOG

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
