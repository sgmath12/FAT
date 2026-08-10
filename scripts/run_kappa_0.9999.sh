#!/bin/bash
# kappa=0.9999 extension (2026-07-18, user q after seeing kappa=0.9 result): pushes the WA
# decay-schedule STARTING point even higher/stronger than the 0.9995 cell already in flight
# (decay glides kappa->1.0 via annealing=(epoch/epochs)^2; kappa=0.9999 = strongest/slowest
# averaging tested so far). Chained AFTER the original {0.9, 0.9995} sweep (waits for
# KAPPA_SWEEP_DONE), and writes a NEW marker (KAPPA_EXT_DONE) so anything chained behind the
# kappa sweep (the AWP-SAM cell) waits for this too, not just the original two.
# Champion bar (kappa 0.999): clean62.75/pgd20 33.96/pgd10 34.18/cw28.41, H(pgd)44.07/H(cw)39.11.
# kappa=0.9 result (done): clean64.03/pgd20 31.73/cw26.80 -- worse than champion on robustness.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/kappa_sweep_chain.log
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

until grep -q "KAPPA_SWEEP_DONE" $LOG 2>/dev/null; do sleep 180; done

k=0.9999
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

echo "KAPPA_EXT_DONE $(date)" >> $LOG
