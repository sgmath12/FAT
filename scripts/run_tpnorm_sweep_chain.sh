#!/bin/bash
# Teacher Lq-norm power sweep (2026-07-06 night), chained behind the centered seed-verify.
# 8 cells: q=2 (L2) x p{-1.01,1,2,4}  +  q=1 (L1) x p{1,2}  +  q=100 (Linf) x p{1,2}.
# (p=-1 exactly is the CLI sentinel -> use -1.01, same convention as the decompw sweep.)
# p=0 control == train_temperature tau16 == existing 41.77 baseline, not re-run.
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
until grep -q "ALL_CENTERED_SEEDVERIFY_DONE" results/CIFAR100/centered_seedverify_chain.log 2>/dev/null; do sleep 60; done
echo "seed-verify done, starting tpnorm sweep $(date)"
for cell in "2 -1.01" "2 1" "2 2" "2 4" "1 1" "1 2" "100 1" "100 2"; do
  set -- $cell; q=$1; p=$2
  $PY main.py --config_name temp_tpnorm.yaml --dataset CIFAR100 --seed 0 --eta "$q" --gamma "$p" \
      > "results/CIFAR100/tpnorm_q${q}_p${p}_driver.log" 2>&1
  echo "tpnorm q${q} p${p} DONE $(date)"
done
echo "ALL_TPNORM_SWEEP_DONE $(date)"
