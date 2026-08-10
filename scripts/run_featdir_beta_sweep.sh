#!/bin/bash
# featdir beta sweep (user, 2026-07-13): beta = head-KL weight in train_feat_direction.
# beta=1.0 already run (featdir H 39.93 / featdir_gainhead H 40.10, both clean+1.5/pgd-2.3 vs
# baseline 41.77). NOTE the detach: beta trains the HEAD ONLY (backbone sees just the direction
# L2 + direction-max attack) -- if this sweep is FLAT, the clean/robust frontier shift is
# backbone/attack-side and the next knobs are tau or a KL/direction hybrid attack, not beta.
# Waits for the temp_gainhead_nodecay run (PID passed as $1, default: poll main.py) to finish.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/featdir_beta_chain.log

while ps aux | grep -q "[m]ain.py"; do sleep 60; done
echo "GPU free, sweep starting $(date)" >> $LOG

for cfg in featdir featdir_gainhead; do
  for b in 0.25 0.5 2.0 4.0; do
    echo "=== $cfg beta $b START $(date) ===" >> $LOG
    $PY -u main.py --config_name $cfg.yaml --dataset CIFAR100 --seed 0 --beta $b \
      > results/CIFAR100/${cfg}_beta${b}_driver.log 2>&1
    echo "=== $cfg beta $b DONE $(date) ===" >> $LOG
  done
done
echo "FEATDIR_BETA_SWEEP_DONE $(date)" >> $LOG
