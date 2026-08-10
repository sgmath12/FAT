#!/bin/bash
# User request 2026-07-18 ~21:50: cancel the kappa {0.995,0.9975} extension, run AWP-SAM first
# (already coded: utils.AdvWeightPerturbSAM, config featdir_k350wa_awpsam.yaml, rho=0.005 matching
# ADR's own gin verbatim), then kappa=0.998 (closer to champion's 0.999 than the 0.995/0.9975
# points already tried). This runs the smoke+full AWP-SAM cell directly (no marker wait), then
# chains kappa=0.998 on the plain champion config right after.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
AWPLOG=results/CIFAR100/awpsam_chain.log
CKDIR_AWP=CIFAR100/checkpoint/featdir_k350wa_awpsam
CKDIR_CHAMP=CIFAR100/checkpoint/featdir_span_random_10step_wa
KLOG=results/CIFAR100/kappa_sweep2_chain.log

echo "=== AWPSAM SMOKE START $(date) ===" >> $AWPLOG
if $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --epochs 1 \
    > results/CIFAR100/awpsam_smoke_driver.log 2>&1; then
  echo "=== AWPSAM smoke OK, full START $(date) ===" >> $AWPLOG
  $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/awpsam_driver.log 2>&1
  cp $CKDIR_AWP/feat_direction_last.pkl $CKDIR_AWP/awpsam_last.pkl 2>/dev/null
  echo "=== AWPSAM DONE (ckpt archived) $(date) ===" >> $AWPLOG
else
  echo "!!! AWPSAM SMOKE FAILED, skipped $(date) ===" >> $AWPLOG
fi
echo "AWPSAM_DONE $(date)" >> $AWPLOG

echo "=== kappa=0.998 START $(date) ===" >> $KLOG
$PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --kappa 0.998 \
  > results/CIFAR100/kappa_0.998_driver.log 2>&1
if [ $? -eq 0 ]; then
  cp $CKDIR_CHAMP/feat_direction_last.pkl $CKDIR_CHAMP/kappa_0.998_last.pkl 2>/dev/null
  echo "=== kappa=0.998 DONE (ckpt archived) $(date) ===" >> $KLOG
else
  echo "!!! kappa=0.998 FAILED $(date) ===" >> $KLOG
fi
echo "KAPPA_SWEEP3_DONE $(date)" >> $KLOG
