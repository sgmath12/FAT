#!/bin/bash
# (2026-07-17) waits for in-flight tau32, does its archive/AA-skip housekeeping, then head_ce
# cell (prioritized ahead of remaining taubeta per user -- low expectation on that grid now),
# then resumes remaining taubeta cells.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

while ps aux | grep -q "[m]ain.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --tau 32"; do sleep 60; done
cp $CKDIR/feat_direction_last.pkl $CKDIR/tauhigh_t32_last.pkl 2>/dev/null
echo "=== tauhigh tau=32 beta=1 DONE (ckpt archived) $(date) ===" >> $LOG
echo "TAUHIGH_DONE $(date)" >> $LOG

echo "=== headce SMOKE START $(date) ===" >> $LOG
if $PY -u main.py --config_name featdir_head_ce.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --epochs 1 \
    > results/CIFAR100/headce_smoke_driver.log 2>&1; then
  echo "=== headce smoke OK, full START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_head_ce.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/headce_driver.log 2>&1
  echo "=== headce DONE $(date) ===" >> $LOG
  $PY -u $EVAL "k350+WA+lamda4+headCE|CIFAR100/checkpoint/featdir_head_ce/feat_direction_last.pkl" >> results/CIFAR100/aa_verify_20260716.log 2>&1
else
  echo "!!! headce SMOKE FAILED $(date)" >> $LOG
fi
echo "HEADCE_DONE $(date)" >> $LOG

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
