#!/bin/bash
# Retry of AWP-SAM (2026-07-18 22:35): the first attempt crashed at epoch5 (awp_warmup) with
# NameError: name '_awp_bn_disable' is not defined -- from utils import * silently drops
# underscore-prefixed names, so methods.py never saw _awp_bn_disable/_awp_bn_enable. Fixed via
# explicit `from utils import _awp_bn_disable, _awp_bn_enable` in methods.py. The smoke gate
# (epochs:1) never caught it because AWP only activates at epoch>=awp_warmup(5); this script
# checks exit code properly this time so a crash can't silently log DONE again.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
AWPLOG=results/CIFAR100/awpsam_chain.log
CKDIR_AWP=CIFAR100/checkpoint/featdir_k350wa_awpsam

WAIT_PID=11743
echo "=== AWPSAM RETRY waiting on PID ${WAIT_PID} (kappa=0.998) $(date) ===" >> $AWPLOG
while kill -0 $WAIT_PID 2>/dev/null; do
  sleep 30
done

echo "=== AWPSAM RETRY SMOKE START $(date) ===" >> $AWPLOG
if $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 --epochs 6 \
    > results/CIFAR100/awpsam_retry_smoke_driver.log 2>&1; then
  echo "=== AWPSAM RETRY smoke OK (6ep, past awp_warmup), full START $(date) ===" >> $AWPLOG
  $PY -u main.py --config_name featdir_k350wa_awpsam.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda 4.0 \
    > results/CIFAR100/awpsam_retry_driver.log 2>&1
  if [ $? -eq 0 ]; then
    cp $CKDIR_AWP/feat_direction_last.pkl $CKDIR_AWP/awpsam_last.pkl 2>/dev/null
    echo "=== AWPSAM RETRY DONE (ckpt archived) $(date) ===" >> $AWPLOG
  else
    echo "!!! AWPSAM RETRY FULL RUN FAILED $(date) ===" >> $AWPLOG
  fi
else
  echo "!!! AWPSAM RETRY SMOKE FAILED (still broken past warmup) $(date) ===" >> $AWPLOG
fi

echo "AWPSAM_RETRY_DONE $(date)" >> $AWPLOG
