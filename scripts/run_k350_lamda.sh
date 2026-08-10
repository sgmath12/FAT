#!/bin/bash
# k350 + lamda cells, 3-step (user, 2026-07-14 12:20): k350 lacks only pgd (-0.7) vs bar and
# already TIES the cw axis (H_cw 37.22 vs 37.29); lamda adds +0.4~0.6 pgd but tends to cost cw
# -> dose {30, 100} to see if the cw-tie survives. Behind ORACLE_RERUN_DONE.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "ORACLE_RERUN_DONE" $LOG 2>/dev/null; do sleep 120; done
for l in 30.0 100.0; do
  echo "=== 3step span k=350 lamda=$l START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda $l \
    > results/CIFAR100/span_k350_l${l}_driver.log 2>&1
  echo "=== 3step span k=350 lamda=$l DONE $(date) ===" >> $LOG
done
echo "K350_LAMDA_DONE $(date)" >> $LOG
