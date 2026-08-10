#!/bin/bash
# eps frontier points behind WA_SEEDCOMPLETE_DONE (2026-07-14 night).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "WA_SEEDCOMPLETE_DONE" $LOG 2>/dev/null; do sleep 120; done
for e in 10 12; do
  echo "=== k350wa eps${e} START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_k350wa_eps${e}.yaml --dataset CIFAR100 --seed 0 --eta 350 \
    > results/CIFAR100/k350wa_eps${e}_driver.log 2>&1
  echo "=== k350wa eps${e} DONE $(date) ===" >> $LOG
done
echo "EPS_FRONTIER_DONE $(date)" >> $LOG
