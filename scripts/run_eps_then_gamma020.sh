#!/bin/bash
# Reordered (user, 2026-07-16 22:35): eps10/eps12 + lamda4 FIRST (higher priority frontier
# points on the current best method), THEN resume the AWP gamma=0.02 pair (baseline g020 was
# cancelled mid-run, restarting clean; k350 g020 never started).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
EVAL=scripts/eval_aa_generic.py

for e in 10 12; do
  cfg="featdir_k350wa_eps${e}"
  echo "=== ${cfg}+lamda4 START $(date) ===" >> $LOG
  $PY -u main.py --config_name ${cfg}.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/${cfg}_lamda4_driver.log 2>&1
  echo "=== ${cfg}+lamda4 DONE $(date) ===" >> $LOG
  $PY -u $EVAL "k350+WA+eps${e}+lamda4|CIFAR100/checkpoint/${cfg}/feat_direction_last.pkl" >> results/CIFAR100/aa_verify_20260716.log 2>&1
done
echo "EPS_LAMDA4_DONE $(date)" >> $LOG

run_and_aa () {
  cfg=$1; label=$2; ckdir=$3; ckfile=$4
  echo "=== $cfg START $(date) ===" >> $LOG
  $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 --eta 350 \
    > results/CIFAR100/${cfg}_driver.log 2>&1
  echo "=== $cfg DONE $(date) ===" >> $LOG
  $PY -u $EVAL "${label}|CIFAR100/checkpoint/${ckdir}/${ckfile}" >> results/CIFAR100/aa_verify_20260716.log 2>&1
}
run_and_aa "temp_baseline_10step_wa_awp_g020" "baseline+WA+AWP gamma0.02" "temp_baseline_10step_wa_awp_g020" "temperature_last.pkl"
run_and_aa "featdir_k350wa_awp_g020"          "k350+WA+AWP gamma0.02"     "featdir_k350wa_awp_g020"          "feat_direction_last.pkl"

echo "AWP_GAMMA_DONE $(date)" >> $LOG
