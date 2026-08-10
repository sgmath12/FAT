#!/bin/bash
# kappa sweep round 2, queued behind the champion-restore rerun (kappa=0.999, PID 3706).
# User request 2026-07-18 19:40: run kappa {0.995, 0.9975} next (0.995 died mid-run last
# time without a diagnosed cause; 0.9975 is a new intermediate point between 0.999 and 0.9995).
# Archives checkpoint immediately after each cell so a later crash can't clobber it.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/kappa_sweep2_chain.log
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step_wa

WAIT_PID=3706
echo "=== KAPPA SWEEP2 waiting on PID ${WAIT_PID} $(date) ===" >> $LOG
while kill -0 $WAIT_PID 2>/dev/null; do
  sleep 30
done
echo "=== PID ${WAIT_PID} (kappa=0.999 rerun) finished $(date), archiving its ckpt ===" >> $LOG
cp $CKDIR/feat_direction_last.pkl $CKDIR/kappa_0.999_rerun_last.pkl 2>/dev/null

for k in 0.995 0.9975; do
  echo "=== kappa=${k} START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --kappa $k \
    > results/CIFAR100/kappa_${k}_driver.log 2>&1
  if [ $? -eq 0 ]; then
    cp $CKDIR/feat_direction_last.pkl $CKDIR/kappa_${k}_last.pkl 2>/dev/null
    echo "=== kappa=${k} DONE (ckpt archived) $(date) ===" >> $LOG
  else
    echo "!!! kappa=${k} FAILED (exit $?) $(date) ===" >> $LOG
  fi
done

echo "KAPPA_SWEEP2_DONE $(date)" >> $LOG
