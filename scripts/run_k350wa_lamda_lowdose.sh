#!/bin/bash
# LOW-DOSE lamda on k350+WA (2026-07-15 15:40): grad-norm diag showed dir-loss backbone grads
# are 5.7x KL's -> effective lamda scale differs 4.2x between pipelines. baseline+WA sweet spot
# lamda1 (cw +0.57!) => k350 equivalent ~4; our tested {10,30,100} skipped it. Cells: lamda {2,4}.
# Behind BATTERY_AA_DONE. Bar: k350+WA lamda0 = 62.61/33.67/cw28.00 (s0).
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
until grep -q "BATTERY_AA_DONE" $LOG 2>/dev/null; do sleep 120; done
for l in 2.0 4.0; do
  echo "=== k350wa lamda $l (lowdose) START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step_wa.yaml --dataset CIFAR100 --seed 0 --eta 350 --lamda $l \
    > results/CIFAR100/k350wa_lamda${l}_driver.log 2>&1
  echo "=== k350wa lamda $l (lowdose) DONE $(date) ===" >> $LOG
done
echo "K350WA_LOWDOSE_DONE $(date)" >> $LOG
