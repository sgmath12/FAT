#!/bin/bash
# kappa (WA EMA decay) sweep on the champion (k350+WA+lamda4, tau16/beta1/eps8, eta350) --
# user q 2026-07-18: does the WA decay knob trade clean for robust? Never swept before
# (always fixed at 0.999 in every WA config). Added --kappa CLI override to utils.py
# load_parser/load_config to make this possible without new yaml files.
# Champion bar (kappa 0.999): clean62.75/pgd20 33.96/pgd10 34.18/cw28.41, H(pgd)44.07/H(cw)39.11.
# kappa=0.9    -> decay starts much lower (0.9->1.0 over training) = weaker/faster-forgetting averaging,
#                 closer to no-WA behavior.
# kappa=0.9995 -> decay starts higher (0.9995->1.0) = stronger/slower averaging than champion.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/kappa_sweep_chain.log
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

echo "=== KAPPA SWEEP START $(date) ===" >> $LOG

for k in 0.9 0.9995; do
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
done

echo "KAPPA_SWEEP_DONE $(date)" >> $LOG
