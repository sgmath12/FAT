#!/bin/bash
# Overnight unattended sweep (user going to sleep, 2026-07-16 23:50), behind AWP_GAMMA_DONE
# (existing eps12+lamda4 -> gamma0.02 pair chain finishes first).
# (1) cons_detach: TRADES-style one-directional consistency vs the bidirectional default,
#     on the champion (k350+WA+lamda4) -- AA'd immediately (direct comparison to champion's AA26.29).
# (2) tau x beta balanced partial grid (8 cells, each tau/beta value appears exactly twice) on
#     the champion pipeline (k350+WA, lamda4, eta350). Champion is tau16/beta1 (bar H(pgd)44.07,
#     H(cw)39.11, AA26.29). Only clean/pgd/cw collected (no AA per cell -- AA the eventual winner
#     after review). Combos: (1,0.25)(1,4)(2,0.5)(2,2)(4,0.25)(4,4)(8,0.5)(8,2).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py
until grep -q "AWP_GAMMA_DONE" $LOG 2>/dev/null; do sleep 180; done

echo "=== consdetach SMOKE START $(date) ===" >> $LOG
if $PY -u main.py --config_name featdir_k350wa_consdetach.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --epochs 1 \
    > results/CIFAR100/consdetach_smoke_driver.log 2>&1; then
  echo "=== consdetach smoke OK, full START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_k350wa_consdetach.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/consdetach_driver.log 2>&1
  echo "=== consdetach DONE $(date) ===" >> $LOG
  $PY -u $EVAL "k350+WA+lamda4+consdetach|CIFAR100/checkpoint/featdir_k350wa_consdetach/feat_direction_last.pkl" >> results/CIFAR100/aa_verify_20260716.log 2>&1
else
  echo "!!! consdetach SMOKE FAILED, skipped $(date) ===" >> $LOG
fi

CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa
TAUBETA=(
  "1 0.25" "1 4" "2 0.5" "2 2" "4 0.25" "4 4" "8 0.5" "8 2"
)
for tb in "${TAUBETA[@]}"; do
  set -- $tb; t=$1; b=$2
  echo "=== taubeta tau=${t} beta=${b} START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --tau $t --beta $b \
    > results/CIFAR100/taubeta_t${t}_b${b}_driver.log 2>&1
  # archive per-cell ckpt (this folder gets overwritten every cell otherwise -- so the eventual
  # cw-winner can still be AA'd later without a retrain)
  cp $CKDIR/feat_direction_last.pkl $CKDIR/taubeta_t${t}_b${b}_last.pkl 2>/dev/null
  echo "=== taubeta tau=${t} beta=${b} DONE (ckpt archived) $(date) ===" >> $LOG
done
echo "TAUBETA_SWEEP_DONE $(date)" >> $LOG
