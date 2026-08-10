#!/bin/bash
# eps x lamda4 stack (user, 2026-07-16 22:30), behind AWP_GAMMA_DONE: push the current best
# (k350+WA+lamda4, clean62.75/AA26.29) toward a more robust-leaning point by adding train_eps.
# eps10 alone (no lamda) gave 58.52/AA26.83; stacking lamda4 should push pgd/cw further while
# landing closer to ADR's clean (57.36). Also eps12+lamda4 for the far end of the frontier.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py
until grep -q "AWP_GAMMA_DONE" $LOG 2>/dev/null; do sleep 120; done

for e in 10 12; do
  cfg="featdir_k350wa_eps${e}"
  echo "=== ${cfg}+lamda4 START $(date) ===" >> $LOG
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/${cfg}_lamda4_driver.log 2>&1
  echo "=== ${cfg}+lamda4 DONE $(date) ===" >> $LOG
  $PY -u $EVAL "k350+WA+eps${e}+lamda4|CIFAR100/checkpoint/${cfg}/feat_direction_last.pkl" >> results/CIFAR100/aa_verify_20260716.log 2>&1
done
echo "EPS_LAMDA4_DONE $(date)" >> $LOG
