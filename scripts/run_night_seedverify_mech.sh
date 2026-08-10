#!/bin/bash
# NIGHT QUEUE 3 (2026-07-14 evening, behind K350_WA_DONE):
# (1) 10-step k350 pure 3-SEED (s0 rerun incl. -- winner ckpt was overwritten) + ckpt ARCHIVE
#     per seed (lesson: ckpts overwrite per config folder). Bar per-seed: 42.18/42.20/42.06.
# (2) mechanism cell 5: featdir_suppress (forbid free subspace; predict robust -> k512 28.91)
# (3) mechanism cell 4: softweight control (= k512 replicate; predict 39.93, AdamW invariance)
PY=/home/seungju/miniforge3/envs/advTrain/bin/python
cd /mnt/d/research/FAT
LOG=results/CIFAR100/night_selfmetric_chain.log
CKDIR=CIFAR100/checkpoint/featdir_span_random_10step
until grep -q "K350_WA_DONE" $LOG 2>/dev/null; do sleep 120; done

for s in 0 1 2; do
  echo "=== 10step k350 seed $s START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_span_random_10step.yaml --dataset CIFAR100 --seed $s --eta 350 \
    > results/CIFAR100/k350_10step_seed${s}_driver.log 2>&1
  cp $CKDIR/feat_direction_last.pkl $CKDIR/k350_seed${s}_last.pkl 2>/dev/null
  cp $CKDIR/feat_direction_best.pkl $CKDIR/k350_seed${s}_best.pkl 2>/dev/null
  echo "=== 10step k350 seed $s DONE (ckpt archived) $(date) ===" >> $LOG
done

echo "=== suppress SMOKE START $(date) ===" >> $LOG
if $PY -u main.py --config_name featdir_suppress.yaml --dataset CIFAR100 --seed 0 --eta 350 --epochs 1 \
    > results/CIFAR100/suppress_smoke_driver.log 2>&1; then
  echo "=== suppress smoke OK, full START $(date) ===" >> $LOG
  $PY -u main.py --config_name featdir_suppress.yaml --dataset CIFAR100 --seed 0 --eta 350 \
    > results/CIFAR100/featdir_suppress_driver.log 2>&1
  echo "=== suppress DONE $(date) ===" >> $LOG
else
  echo "!!! suppress SMOKE FAILED $(date)" >> $LOG
fi

echo "=== softweight START $(date) ===" >> $LOG
$PY -u main.py --config_name featdir_softweight.yaml --dataset CIFAR100 --seed 0 \
  > results/CIFAR100/featdir_softweight_driver.log 2>&1
echo "NIGHT3_DONE $(date)" >> $LOG
